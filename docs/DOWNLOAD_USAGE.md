# Ensembl Sequence Downloader Usage Guide

This guide explains how to use the standalone `scripts/download_sequences.py` script to download DNA sequences from the Ensembl REST API for use in the MotifResearch pipeline.

## Overview

The downloader script (`scripts/download_sequences.py`) fetches DNA sequences from Ensembl based on genomic coordinates. It's designed to run locally and cache sequences for later use in the notebook pipeline.

### Key Features

- **BED file parsing**: Supports standard BED format and UCSC table format (with bin column)
- **Coordinate conversion**: Automatically converts BED 0-based coordinates to Ensembl 1-based
- **Species-specific handling**: Special support for chicken galGal4 (archive API) and zebrafish danRer10 (GRCz10)
- **Smart retry logic**: Exponential backoff for rate limiting (429) and service errors (503)
- **Parallel downloads**: Chunked batch processing with configurable workers
- **Quality filtering**: Removes sequences containing 'N', standardizes length
- **JSONL output**: Easy-to-use format for notebook integration

## Installation

No additional dependencies required beyond the base requirements:

```bash
# All dependencies are in requirements.txt
pip install -r requirements.txt
```

## Basic Usage

### Download from a BED file

```bash
python scripts/download_sequences.py \
    --species human \
    --bed data/raw/human_enhancers.bed \
    --output data/processed/human_sequences.jsonl
```

### Download from coordinate strings

```bash
python scripts/download_sequences.py \
    --species mouse \
    --coords "chr1:1000-1200,chr2:2000-2200,chr3:5000-5150" \
    --output data/processed/mouse_sequences.jsonl
```

## Species-Specific Examples

### Human (hg38 - default)

```bash
python scripts/download_sequences.py \
    --species human \
    --bed data/raw/human_enhancers.bed \
    --output data/processed/human_sequences.jsonl \
    --max_sequences 30000 \
    --target_length 200
```

### Mouse (mm10 - default)

```bash
python scripts/download_sequences.py \
    --species mouse \
    --bed data/raw/mouse_enhancers.bed \
    --output data/processed/mouse_sequences.jsonl \
    --max_sequences 20000 \
    --target_length 200
```

### Zebrafish (danRer10 - requires assembly specification)

```bash
python scripts/download_sequences.py \
    --species zebrafish \
    --assembly danRer10 \
    --bed data/raw/zebrafish_enhancers.bed \
    --output data/processed/zebrafish_sequences.jsonl \
    --max_sequences 15000 \
    --target_length 200
```

### Chicken (galGal4 - requires assembly specification, uses archive API)

```bash
python scripts/download_sequences.py \
    --species chicken \
    --assembly galGal4 \
    --bed data/raw/chicken_enhancers.bed \
    --output data/processed/chicken_sequences.jsonl \
    --max_sequences 15000 \
    --target_length 200
```

## Recommended Parameters

Based on testing and Ensembl API rate limits:

- **chunk_size**: `15` (default) - Number of sequences per download chunk
- **max_workers**: `8` (default) - Number of parallel download threads
- **target_length**: `200` (default) - Standard sequence length for the pipeline
- **max_retries**: `3` (default) - Retry attempts for transient errors
- **max_sequences**: Adjust based on your needs (30000 for human, 15000 for chicken, etc.)

### Performance Tuning

For **faster downloads** (if you have good network and want to push limits):
```bash
--chunk_size 20 --max_workers 10
```

For **more conservative** rate limiting (if experiencing 429 errors):
```bash
--chunk_size 10 --max_workers 4
```

## Command Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--species` | Yes | Species name: human, mouse, zebrafish, chicken | - |
| `--bed` | * | Path to BED file with coordinates | - |
| `--coords` | * | Comma-separated coordinate strings | - |
| `--output` | Yes | Output JSONL file path | - |
| `--assembly` | No | Assembly version (danRer10, galGal4, etc.) | Latest |
| `--max_sequences` | No | Maximum number of sequences to download | 50000 |
| `--target_length` | No | Target sequence length (trim/pad) | 200 |
| `--max_retries` | No | Maximum retry attempts per sequence | 3 |
| `--chunk_size` | No | Sequences per chunk | 15 |
| `--max_workers` | No | Parallel download workers | 8 |

\* Either `--bed` or `--coords` must be specified (but not both)

## Output Format

The script generates JSONL (JSON Lines) files with one sequence per line:

```json
{"chrom": "chr1", "start": 123456, "end": 123656, "sequence": "ATCG..."}
{"chrom": "chr2", "start": 234567, "end": 234767, "sequence": "GCTA..."}
```

Each object contains:
- `chrom`: Chromosome name (as in BED file)
- `start`: Start position (0-based, as in BED file)
- `end`: End position (exclusive, as in BED file)
- `sequence`: DNA sequence string (uppercase, length = target_length)

## Integration with Notebook Pipeline

### File Placement

Place the generated JSONL files in the expected locations:

```
data/processed/human_sequences.jsonl
data/processed/mouse_sequences.jsonl
data/processed/zebrafish_sequences.jsonl
data/processed/chicken_sequences.jsonl
```

### Notebook Auto-Detection

The updated `notebooks/main_pipeline.ipynb` automatically detects and uses these cached files:

1. **In Colab**: Checks `{PROJECT_DIR}/data/processed/{species}_sequences.jsonl`
2. **Locally**: Same location, plus prints recommendation to use standalone script
3. **Fallback**: If cache missing, downloads via API (with improved logic)

### Running the Full Pipeline

1. **Download sequences locally** (recommended for large datasets):
   ```bash
   # Download all species
   python scripts/download_sequences.py --species human --bed data/raw/human.bed --output data/processed/human_sequences.jsonl
   python scripts/download_sequences.py --species mouse --bed data/raw/mouse.bed --output data/processed/mouse_sequences.jsonl
   python scripts/download_sequences.py --species zebrafish --assembly danRer10 --bed data/raw/zebrafish.bed --output data/processed/zebrafish_sequences.jsonl
   python scripts/download_sequences.py --species chicken --assembly galGal4 --bed data/raw/chicken.bed --output data/processed/chicken_sequences.jsonl
   ```

2. **Run the notebook**:
   - The notebook will detect the cached files and skip downloads
   - Sequences are loaded from JSONL instead of re-downloading

## BED File Format

The script supports both standard BED and UCSC table formats:

### Standard BED format
```
chr1    123456    123656    region1    100    +
chr2    234567    234767    region2    200    -
```

### UCSC table format (with bin column)
```
585    chr1    123456    123656    region1    100    +
586    chr2    234567    234767    region2    200    -
```

The script automatically detects the format and parses accordingly.

## Coordinate Filtering

The script automatically filters regions to **150-250 bp** length range:

- Regions shorter than 150 bp are skipped
- Regions longer than 250 bp are skipped
- Sequences with 'N' bases are filtered out
- Downloaded sequences are standardized to `target_length` (trim/pad with 'A')

## Troubleshooting

### Rate Limiting (429 errors)

If you see frequent 429 errors:
```bash
# Reduce chunk size and workers
python scripts/download_sequences.py ... --chunk_size 10 --max_workers 4
```

### Service Unavailable (503 errors)

The script automatically retries with exponential backoff. If persistent:
- Wait a few minutes and retry
- Check Ensembl status: https://www.ensembl.org/

### No sequences downloaded

Check:
1. BED file format is correct
2. Coordinates are in the valid range (150-250 bp)
3. Chromosome names match assembly (e.g., 'chr1' vs '1')
4. Assembly parameter is correct for your species

### Slow downloads

Expected download times (at ~15 requests/second):
- 1,000 sequences: ~1-2 minutes
- 10,000 sequences: ~10-15 minutes
- 30,000 sequences: ~30-45 minutes

Increase `--max_workers` (up to 10) for faster downloads if network allows.

## Advanced Usage

### Custom sequence length

```bash
# Download 250 bp sequences
python scripts/download_sequences.py \
    --species human \
    --bed data/raw/human.bed \
    --output data/processed/human_250bp.jsonl \
    --target_length 250
```

### Limit number of sequences

```bash
# Download only 1000 sequences for testing
python scripts/download_sequences.py \
    --species mouse \
    --bed data/raw/mouse.bed \
    --output data/processed/mouse_test.jsonl \
    --max_sequences 1000
```

### Use specific assembly

```bash
# Download from older assembly version
python scripts/download_sequences.py \
    --species zebrafish \
    --assembly danRer10 \
    --bed data/raw/zebrafish_danRer10.bed \
    --output data/processed/zebrafish_sequences.jsonl
```

## Notes

- **BED coordinates are 0-based**: The script automatically converts to Ensembl 1-based (start + 1)
- **Chromosome naming**: Script removes 'chr' prefix for Ensembl API compatibility
- **Sequence standardization**: All sequences are trimmed/padded to exact target_length
- **Quality control**: Sequences containing 'N' bases are automatically filtered out
- **Rate limiting**: Built-in delays and chunking respect Ensembl API limits
- **Caching**: Generate files once and reuse them across multiple notebook runs

## Support

For issues or questions:
- Check the script's help: `python scripts/download_sequences.py --help`
- Review Ensembl REST API documentation: https://rest.ensembl.org/
- Open an issue in the repository
