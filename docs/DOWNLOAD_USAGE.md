# Prefetching Ensembl sequences locally (recommended)

Why: Downloading many sequences via the Ensembl REST API from within the notebook is slow and will waste GPU time on A100. Instead, run the downloader locally (on your PC) and save JSONL files. The notebook detects these JSONL caches and will skip re-downloading.

Where to put outputs:
- Place JSONL files in: <PROJECT_DIR>/data/processed/
- Filenames must be: `{species}_sequences.jsonl` (e.g., `human_sequences.jsonl`, `mouse_sequences.jsonl`)

Recommended usage (examples)
1. Download human sequences from a BED file:
   ```
   python scripts/download_sequences.py \
     --species human \
     --bed /path/to/GRCh38-cCRES.bed \
     --output data/processed/human_sequences.jsonl \
     --max_sequences 30000 \
     --chunk_size 15 \
     --max_workers 8
   ```

2. Download mouse sequences from coordinates JSON:
   ```
   python scripts/download_sequences.py \
     --species mouse \
     --coords my_mouse_coords.json \
     --output data/processed/mouse_sequences.jsonl \
     --max_sequences 15000
   ```

Recommended parameters
- chunk_size: 15 (keeps requests ≈15/s)
- max_workers: 6–12 depending on your network/CPU
- target_length: 200 (matches notebook)
- max_retries: 3

Notes & best practices
- The script converts BED 0-based starts to Ensembl 1-based coordinates (start + 1), avoiding off-by-one errors.
- The script filters out sequences with 'N' and standardizes length by trimming or padding with 'A'.
- If you need even faster local performance: download the genome FASTA and extract sequences locally from the BED (I can add a helper script for that).
- Be polite to Ensembl: avoid running too many concurrent downloads. The defaults are tuned to be friendly.
