#!/usr/bin/env python3
"""
scripts/download_sequences.py

Standalone downloader to fetch sequences for coordinates (BED) or for a list of coordinates,
with improved Ensembl requests (start+1 fix, backoff, chunked rate-limiting) and caching.

Usage examples:
  # Download from a BED file
  python scripts/download_sequences.py --species human --bed /path/to/GRCh38-cCRES.bed \
    --output data/processed/human_sequences.jsonl --max_sequences 30000

  # Download using coordinates JSON (list of dicts with chrom,start,end)
  python scripts/download_sequences.py --species mouse --coords coords.json \
    --output data/processed/mouse_sequences.jsonl --max_sequences 15000

This script writes newline-delimited JSON (JSONL). Each line:
  {"chrom": "chr1", "start": 12345, "end": 12545, "sequence": "ACTG..."}

"""
import argparse
import json
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import requests
from tqdm import tqdm

# Defaults
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) enhancer-repair-downloader/1.0"
DEFAULT_TIMEOUT = 15
CHUNK_SIZE = 15  # number of requests per second (Ensembl-friendly)
DEFAULT_MAX_WORKERS = 8

SPECIES_MAP = {
    "human": "homo_sapiens",
    "mouse": "mus_musculus",
    "zebrafish": "danio_rerio",
    "chicken": "gallus_gallus",
}

def parse_bed(filepath: Path, max_sequences: int = 50000) -> List[Dict]:
    coords = []
    if not filepath.exists():
        raise FileNotFoundError(f"BED file not found: {filepath}")
    with filepath.open("r") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            # Support UCSC table (bin, chrom, start, end, ...)
            if len(parts) >= 4 and parts[0].isdigit() and parts[1].startswith("chr"):
                chrom = parts[1]
                start = int(parts[2])
                end = int(parts[3])
            else:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
            length = end - start
            if 150 <= length <= 250:
                coords.append({"chrom": chrom, "start": start, "end": end})
            if len(coords) >= max_sequences * 2:
                break
    if len(coords) > max_sequences:
        random.seed(42)
        coords = random.sample(coords, max_sequences)
    return coords

def build_ensembl_url(chrom: str, start: int, end: int, species: str, assembly: Optional[str] = None):
    # convert BED 0-based start to Ensembl 1-based inclusive
    start1 = start + 1
    species_name = SPECIES_MAP.get(species, SPECIES_MAP["human"])
    base_url = "https://rest.ensembl.org"
    # special-case chicken galGal4 to use archive
    if species == "chicken" and assembly == "galGal4":
        base_url = "https://e78.rest.ensembl.org"
    url = f"{base_url}/sequence/region/{species_name}/{chrom.replace('chr','')}:{start1}-{end}?content-type=text/plain"
    # special zebrafish assembly parameter
    if species == "zebrafish" and assembly == "danRer10":
        url += "&coord_system_version=GRCz10"
    return url

def fetch_one(session: requests.Session, coord: Dict, species: str, assembly: Optional[str], target_length: int, max_retries: int):
    chrom = coord["chrom"]
    start = coord["start"]
    end = coord["end"]
    url = build_ensembl_url(chrom, start, end, species, assembly)
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 200:
                seq = r.text.strip().upper()
                # basic validation
                if len(seq) >= 150 and "N" not in seq:
                    # standardize length
                    if len(seq) > target_length:
                        seq = seq[:target_length]
                    elif len(seq) < target_length:
                        seq = seq + "A" * (target_length - len(seq))
                    return {"chrom": chrom, "start": start, "end": end, "sequence": seq}
                else:
                    return None
            elif r.status_code in (429, 503):
                # rate limited or service unavailable -> backoff and retry
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                # other client/server errors -> do a backoff retry
                time.sleep(backoff)
                backoff *= 2
                continue
        except requests.RequestException:
            time.sleep(backoff)
            backoff *= 2
            continue
    return None

def download_coordinates(coords: List[Dict], species: str, assembly: Optional[str], output_path: Path,
                         target_length: int = 200, max_retries: int = 3, chunk_size: int = CHUNK_SIZE,
                         max_workers: int = DEFAULT_MAX_WORKERS):
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    results = []
    seen = 0
    total = len(coords)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as out_f:
        # chunked processing: chunk_size requests per second, sleep 1 second between chunks
        for i in range(0, total, chunk_size):
            batch = coords[i:i+chunk_size]
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for coord in batch:
                    futures.append(exe.submit(fetch_one, session, coord, species, assembly, target_length, max_retries))
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        out_f.write(json.dumps(res) + "\n")
                        seen += 1
            # be polite to Ensembl: wait ~1s after each chunk
            time.sleep(1.0)
    return seen

def load_coords_json(path: Path, max_sequences: Optional[int] = None) -> List[Dict]:
    with path.open("r") as fh:
        coords = json.load(fh)
    if max_sequences and len(coords) > max_sequences:
        random.seed(42)
        coords = random.sample(coords, max_sequences)
    return coords

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", required=True, choices=list(SPECIES_MAP.keys()))
    p.add_argument("--bed", type=Path, help="Input BED file with coordinates")
    p.add_argument("--coords", type=Path, help="Input JSON file with coordinates: list of {chrom,start,end}")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL file path")
    p.add_argument("--assembly", type=str, default=None, help="Assembly hint (e.g., danRer10, galGal4)")
    p.add_argument("--max_sequences", type=int, default=50000)
    p.add_argument("--target_length", type=int, default=200)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    p.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = p.parse_args()

    if not args.bed and not args.coords:
        p.error("Either --bed or --coords must be provided")

    if args.bed:
        coords = parse_bed(args.bed, max_sequences=args.max_sequences)
    else:
        coords = load_coords_json(args.coords, max_sequences=args.max_sequences)

    if len(coords) == 0:
        print("No coordinates to download. Exiting.")
        return

    print(f"Starting download: species={args.species}, coords={len(coords):,}, output={args.output}")
    start_time = time.time()
    n_downloaded = download_coordinates(coords, args.species, args.assembly, args.output,
                                        target_length=args.target_length,
                                        max_retries=args.max_retries,
                                        chunk_size=args.chunk_size,
                                        max_workers=args.max_workers)
    elapsed = time.time() - start_time
    print(f"Done. Downloaded {n_downloaded:,} sequences in {elapsed/60:.1f} minutes. Saved to {args.output}")

if __name__ == "__main__":
    main()
