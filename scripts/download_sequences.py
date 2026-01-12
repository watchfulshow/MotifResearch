#!/usr/bin/env python3
"""
Standalone Ensembl sequence downloader with BED coordinate support.

This script downloads DNA sequences from the Ensembl REST API based on genomic
coordinates from BED files or direct coordinate specifications. It handles:
- BED file parsing (including UCSC table format with bin column)
- Region filtering (150-250 bp length range)
- BED 0-based to Ensembl 1-based coordinate conversion (start + 1)
- Species-specific assembly handling
- Exponential backoff for rate limiting (429, 503 errors)
- Chunked batch downloads with parallel workers
- Sequence filtering and standardization
- JSONL output format

Usage:
    python download_sequences.py --species human --bed data/raw/human_enhancers.bed --output data/processed/human_sequences.jsonl
    python download_sequences.py --species zebrafish --assembly danRer10 --coords chr1:1000-1200,chr2:2000-2200 --output zebrafish.jsonl
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error


def parse_bed_file(filepath: str, max_sequences: int = 50000) -> List[Dict[str, any]]:
    """
    Parse BED file and extract coordinates.
    
    Supports both standard BED format and UCSC table format (with bin column).
    Filters regions to 150-250 bp length range.
    
    Args:
        filepath: Path to BED file
        max_sequences: Maximum number of sequences to return
        
    Returns:
        List of coordinate dictionaries with keys: chrom, start, end, length
    """
    coordinates = []
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: BED file not found: {filepath}", file=sys.stderr)
        return []
    
    print(f"Loading coordinates from {filepath}...")
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Skip header/comment lines
                if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                # Handle UCSC table format: bin, chrom, start, end, ...
                # UCSC format has bin (integer) in first column and chrom in second
                if len(parts) >= 4 and parts[0].isdigit() and parts[1].startswith('chr'):
                    chrom = parts[1]
                    start = int(parts[2])
                    end = int(parts[3])
                else:
                    # Standard BED format: chrom, start, end, ...
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                
                length = end - start
                
                # Filter for desired length range
                if 150 <= length <= 250:
                    coordinates.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'length': length
                    })
                
                # Stop early if we have enough candidates
                if len(coordinates) >= max_sequences * 2:
                    break
        
        # Random sample if we have more than needed
        if len(coordinates) > max_sequences:
            random.seed(42)
            coordinates = random.sample(coordinates, max_sequences)
        
        print(f"✓ Loaded {len(coordinates):,} coordinates (filtered to 150-250 bp)")
        return coordinates
        
    except Exception as e:
        print(f"Error reading BED file: {e}", file=sys.stderr)
        return []


def parse_coords_string(coords: str) -> List[Dict[str, any]]:
    """
    Parse coordinate string like 'chr1:1000-1200,chr2:2000-2200'.
    
    Args:
        coords: Comma-separated coordinate strings
        
    Returns:
        List of coordinate dictionaries
    """
    coordinates = []
    
    for coord_str in coords.split(','):
        coord_str = coord_str.strip()
        if ':' not in coord_str or '-' not in coord_str:
            print(f"Warning: Invalid coordinate format: {coord_str}", file=sys.stderr)
            continue
        
        try:
            chrom_part, range_part = coord_str.split(':')
            start_str, end_str = range_part.split('-')
            
            chrom = chrom_part.strip()
            start = int(start_str.strip())
            end = int(end_str.strip())
            length = end - start
            
            if 150 <= length <= 250:
                coordinates.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'length': length
                })
            else:
                print(f"Warning: Coordinate {coord_str} outside 150-250 bp range, skipping", file=sys.stderr)
        
        except Exception as e:
            print(f"Error parsing coordinate {coord_str}: {e}", file=sys.stderr)
    
    print(f"✓ Parsed {len(coordinates):,} coordinates from string")
    return coordinates


def build_ensembl_url(chrom: str, start: int, end: int, species: str, assembly: Optional[str] = None) -> str:
    """
    Build Ensembl REST API URL for sequence retrieval.
    
    Converts BED 0-based coordinates to Ensembl 1-based by adding 1 to start.
    Handles species-specific assembly requirements.
    
    Args:
        chrom: Chromosome name
        start: Start position (0-based BED format)
        end: End position (exclusive in BED format)
        species: Species name (human, mouse, zebrafish, chicken)
        assembly: Assembly version (optional, required for older assemblies)
        
    Returns:
        Ensembl REST API URL
    """
    species_map = {
        'human': 'homo_sapiens',
        'mouse': 'mus_musculus',
        'zebrafish': 'danio_rerio',
        'chicken': 'gallus_gallus'
    }
    
    # Convert BED 0-based start to Ensembl 1-based
    ensembl_start = start + 1
    ensembl_end = end  # End is already correct for Ensembl (inclusive)
    
    # Use archive API for chicken galGal4 assembly
    if species == 'chicken' and assembly == 'galGal4':
        base_url = "https://e78.rest.ensembl.org"
    else:
        base_url = "https://rest.ensembl.org"
    
    species_name = species_map.get(species, 'homo_sapiens')
    chrom_clean = chrom.replace('chr', '')
    
    url = (f"{base_url}/sequence/region/"
           f"{species_name}/{chrom_clean}:{ensembl_start}-{ensembl_end}?"
           f"content-type=text/plain")
    
    # Add coord_system_version for zebrafish danRer10
    if species == 'zebrafish' and assembly == 'danRer10':
        url += "&coord_system_version=GRCz10"
    
    return url


def fetch_sequence_with_backoff(url: str, max_retries: int = 3) -> Optional[str]:
    """
    Fetch sequence from URL with exponential backoff for transient errors.
    
    Retries on 429 (Too Many Requests) and 503 (Service Unavailable) with
    exponential backoff. Other errors fail immediately.
    
    Args:
        url: Ensembl REST API URL
        max_retries: Maximum number of retry attempts
        
    Returns:
        DNA sequence string or None if failed
    """
    backoff_time = 1.0  # Start with 1 second
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'MotifResearch/1.0 (Ensembl sequence downloader)'
            })
            
            with urllib.request.urlopen(req, timeout=10) as response:
                sequence = response.read().decode('utf-8').strip().upper()
                return sequence
                
        except urllib.error.HTTPError as e:
            # Exponential backoff for rate limiting and service unavailable
            if e.code in (429, 503):
                if attempt < max_retries - 1:
                    print(f"  Rate limit/service error (HTTP {e.code}), retrying in {backoff_time:.1f}s...", file=sys.stderr)
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                    continue
                else:
                    print(f"  Failed after {max_retries} retries (HTTP {e.code})", file=sys.stderr)
                    return None
            else:
                # Other HTTP errors fail immediately
                print(f"  HTTP error {e.code}: {e.reason}", file=sys.stderr)
                return None
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff_time)
                backoff_time *= 2
                continue
            else:
                print(f"  Error: {e}", file=sys.stderr)
                return None
    
    return None


def download_chunk(coords: List[Dict], species: str, assembly: Optional[str], 
                   target_length: int, max_retries: int) -> List[Dict]:
    """
    Download a chunk of sequences (single-threaded within chunk).
    
    Args:
        coords: List of coordinates to download
        species: Species name
        assembly: Assembly version
        target_length: Target sequence length
        max_retries: Maximum retry attempts per sequence
        
    Returns:
        List of sequence dictionaries
    """
    sequences = []
    
    for coord in coords:
        url = build_ensembl_url(coord['chrom'], coord['start'], coord['end'], species, assembly)
        seq = fetch_sequence_with_backoff(url, max_retries)
        
        if seq and 'N' not in seq and len(seq) >= 150:
            # Standardize to target length
            if len(seq) > target_length:
                seq = seq[:target_length]
            elif len(seq) < target_length:
                seq = seq + 'A' * (target_length - len(seq))
            
            sequences.append({
                'chrom': coord['chrom'],
                'start': coord['start'],
                'end': coord['end'],
                'sequence': seq
            })
    
    return sequences


def download_sequences(coordinates: List[Dict], species: str, assembly: Optional[str] = None,
                      target_length: int = 200, max_retries: int = 3,
                      chunk_size: int = 15, max_workers: int = 8) -> List[Dict]:
    """
    Download sequences in chunked batches with parallel workers.
    
    Splits coordinates into chunks and downloads each chunk in parallel using
    ThreadPoolExecutor. Rate limiting is achieved through chunk_size (default 15
    sequences per chunk) and adding delays between chunks.
    
    Args:
        coordinates: List of coordinate dictionaries
        species: Species name
        assembly: Assembly version (optional)
        target_length: Target sequence length (default: 200)
        max_retries: Maximum retry attempts (default: 3)
        chunk_size: Number of sequences per chunk (default: 15)
        max_workers: Number of parallel workers (default: 8)
        
    Returns:
        List of sequence dictionaries with chrom, start, end, sequence
    """
    print(f"\nDownloading {len(coordinates):,} sequences from Ensembl API")
    print(f"Species: {species}, Assembly: {assembly or 'default'}")
    print(f"Chunk size: {chunk_size}, Max workers: {max_workers}")
    print(f"Estimated time: {len(coordinates) / chunk_size / max_workers * 1.5:.1f} minutes")
    
    # Split into chunks
    chunks = [coordinates[i:i + chunk_size] for i in range(0, len(coordinates), chunk_size)]
    
    all_sequences = []
    failed_count = 0
    
    # Download chunks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(download_chunk, chunk, species, assembly, target_length, max_retries): i
            for i, chunk in enumerate(chunks)
        }
        
        # Process completed chunks
        completed = 0
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_sequences = future.result()
                all_sequences.extend(chunk_sequences)
                completed += 1
                
                # Progress update
                progress = completed / len(chunks) * 100
                print(f"Progress: {completed}/{len(chunks)} chunks ({progress:.1f}%) - "
                      f"{len(all_sequences):,} sequences downloaded", end='\r')
                
                # Rate limiting between chunks
                if completed < len(chunks):
                    time.sleep(1.0 / max_workers)
                    
            except Exception as e:
                print(f"\nError processing chunk {chunk_idx}: {e}", file=sys.stderr)
                failed_count += len(chunks[chunk_idx])
    
    print()  # New line after progress
    print(f"✓ Downloaded {len(all_sequences):,} sequences (failed/filtered: {len(coordinates) - len(all_sequences):,})")
    
    return all_sequences


def write_jsonl(sequences: List[Dict], output_path: str):
    """
    Write sequences to JSONL file.
    
    Args:
        sequences: List of sequence dictionaries
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for seq_dict in sequences:
            f.write(json.dumps(seq_dict) + '\n')
    
    print(f"✓ Wrote {len(sequences):,} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download DNA sequences from Ensembl REST API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download human enhancers from BED file
  python download_sequences.py --species human --bed data/raw/human_enhancers.bed --output data/processed/human_sequences.jsonl
  
  # Download zebrafish sequences with specific assembly
  python download_sequences.py --species zebrafish --assembly danRer10 --bed data/raw/zebrafish.bed --output zebrafish.jsonl
  
  # Download from coordinate string
  python download_sequences.py --species mouse --coords "chr1:1000-1200,chr2:2000-2200" --output mouse.jsonl
  
  # Customize download parameters
  python download_sequences.py --species chicken --assembly galGal4 --bed chicken.bed --output chicken.jsonl --chunk_size 10 --max_workers 4
        """
    )
    
    parser.add_argument('--species', required=True, 
                       choices=['human', 'mouse', 'zebrafish', 'chicken'],
                       help='Species name')
    parser.add_argument('--bed', type=str,
                       help='Path to BED file with coordinates')
    parser.add_argument('--coords', type=str,
                       help='Comma-separated coordinates (e.g., chr1:1000-1200,chr2:2000-2200)')
    parser.add_argument('--output', required=True, type=str,
                       help='Output JSONL file path')
    parser.add_argument('--assembly', type=str,
                       help='Assembly version (e.g., danRer10, galGal4)')
    parser.add_argument('--max_sequences', type=int, default=50000,
                       help='Maximum number of sequences to download (default: 50000)')
    parser.add_argument('--target_length', type=int, default=200,
                       help='Target sequence length (default: 200)')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum retry attempts per sequence (default: 3)')
    parser.add_argument('--chunk_size', type=int, default=15,
                       help='Number of sequences per chunk (default: 15)')
    parser.add_argument('--max_workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.bed and not args.coords:
        parser.error("Either --bed or --coords must be specified")
    
    if args.bed and args.coords:
        parser.error("Cannot specify both --bed and --coords")
    
    # Load coordinates
    if args.bed:
        coordinates = parse_bed_file(args.bed, args.max_sequences)
    else:
        coordinates = parse_coords_string(args.coords)
        if len(coordinates) > args.max_sequences:
            random.seed(42)
            coordinates = random.sample(coordinates, args.max_sequences)
    
    if not coordinates:
        print("Error: No valid coordinates found", file=sys.stderr)
        sys.exit(1)
    
    # Download sequences
    sequences = download_sequences(
        coordinates,
        args.species,
        args.assembly,
        args.target_length,
        args.max_retries,
        args.chunk_size,
        args.max_workers
    )
    
    if not sequences:
        print("Error: No sequences downloaded", file=sys.stderr)
        sys.exit(1)
    
    # Write output
    write_jsonl(sequences, args.output)
    
    print("\n✓ Download complete!")


if __name__ == '__main__':
    main()
