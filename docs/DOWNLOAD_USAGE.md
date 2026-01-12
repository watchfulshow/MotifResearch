# Local Sequence Downloader Usage Guide

This guide explains how to use the standalone `scripts/download_sequences.py` script to pre-download genomic sequences locally, avoiding wasted A100 GPU time in Google Colab.

## Why Use the Local Downloader?

- **Save GPU time**: Download sequences on your local machine before running the notebook on expensive GPU instances
- **Better reliability**: Exponential backoff and retry logic handles transient network errors
- **Faster downloads**: Chunked parallel downloads (8 workers by default) speed up the process
- **Respects Ensembl**: Rate-limited to ~15 requests/second to avoid overwhelming the Ensembl API
- **Proper coordinate handling**: Converts BED 0-based coordinates to Ensembl 1-based format (start + 1)

## Installation

First, ensure you have the required dependencies:

```bash
pip install requests tqdm
```

## Basic Usage

The script supports two input methods:

### 1. Download from a BED file

```bash
python scripts/download_sequences.py \
  --species human \
  --bed /path/to/coordinates.bed \
  --output data/processed/human_sequences.jsonl \
  --max_sequences 30000
```

### 2. Download from a JSON coordinates file

```bash
python scripts/download_sequences.py \
  --species mouse \
  --coords coordinates.json \
  --output data/processed/mouse_sequences.jsonl \
  --max_sequences 15000
```

The coordinates JSON file should contain a list of dictionaries with `chrom`, `start`, and `end` keys:
```json
[
  {"chrom": "chr1", "start": 12345, "end": 12545},
  {"chrom": "chr2", "start": 67890, "end": 68090}
]
```

## Species-Specific Examples

### Human (GRCh38)

```bash
python scripts/download_sequences.py \
  --species human \
  --bed data/raw/GRCh38-cCREs.bed \
  --output data/processed/human_sequences.jsonl \
  --max_sequences 30000 \
  --target_length 200 \
  --chunk_size 15 \
  --max_workers 8
```

### Mouse (mm10)

```bash
python scripts/download_sequences.py \
  --species mouse \
  --bed data/raw/mm10-cCREs.bed \
  --output data/processed/mouse_sequences.jsonl \
  --max_sequences 20000 \
  --target_length 200 \
  --chunk_size 15 \
  --max_workers 8
```

### Zebrafish (danRer10/GRCz10)

```bash
python scripts/download_sequences.py \
  --species zebrafish \
  --bed data/raw/zebrafish-enhancers.bed \
  --output data/processed/zebrafish_sequences.jsonl \
  --assembly danRer10 \
  --max_sequences 15000 \
  --target_length 200 \
  --chunk_size 15 \
  --max_workers 8
```

### Chicken (galGal4)

```bash
python scripts/download_sequences.py \
  --species chicken \
  --bed data/raw/chicken-enhancers.bed \
  --output data/processed/chicken_sequences.jsonl \
  --assembly galGal4 \
  --max_sequences 15000 \
  --target_length 200 \
  --chunk_size 15 \
  --max_workers 8
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--species` | Yes | - | Species to download: `human`, `mouse`, `zebrafish`, or `chicken` |
| `--bed` | One of `--bed` or `--coords` | - | Path to BED file with coordinates |
| `--coords` | One of `--bed` or `--coords` | - | Path to JSON file with coordinates |
| `--output` | Yes | - | Path for output JSONL file |
| `--assembly` | No | - | Assembly version (e.g., `danRer10` for zebrafish, `galGal4` for chicken) |
| `--max_sequences` | No | 50000 | Maximum number of sequences to download |
| `--target_length` | No | 200 | Target sequence length (trim/pad to this length) |
| `--max_retries` | No | 3 | Maximum retry attempts per sequence |
| `--chunk_size` | No | 15 | Number of requests per chunk (rate-limiting) |
| `--max_workers` | No | 8 | Number of parallel download threads |

## Recommended Settings

For optimal performance while being respectful to Ensembl:

- **chunk_size**: 15 (15 requests per second)
- **max_workers**: 8 (8 parallel threads)
- **target_length**: 200 (standard enhancer length)
- **max_retries**: 3 (retry failed requests)

## Output Format

The script generates a newline-delimited JSON (JSONL) file. Each line contains one sequence record:

```json
{"chrom": "chr1", "start": 12345, "end": 12545, "sequence": "ACTGATCG..."}
{"chrom": "chr2", "start": 67890, "end": 68090, "sequence": "TGCATGCA..."}
```

## File Placement for Notebook

To ensure the notebook automatically detects and uses your pre-downloaded sequences:

1. Place the JSONL files in the `data/processed/` directory
2. Use this naming convention: `{species}_sequences.jsonl`
   - Human: `data/processed/human_sequences.jsonl`
   - Mouse: `data/processed/mouse_sequences.jsonl`
   - Zebrafish: `data/processed/zebrafish_sequences.jsonl`
   - Chicken: `data/processed/chicken_sequences.jsonl`

When running the notebook, it will check for these files and load sequences from cache instead of re-downloading.

## Best Practices

1. **Run one species at a time**: Don't run multiple downloads simultaneously to avoid overwhelming Ensembl
2. **Monitor progress**: The script shows a progress bar and prints a summary when complete
3. **Check success rate**: Review the final count to ensure most sequences downloaded successfully
4. **Verify output**: Check the JSONL file size and content before uploading to Colab
5. **Use appropriate max_sequences**: Match the dataset sizes to your GPU type:
   - A100: 30k human, 20k mouse, 15k zebrafish/chicken
   - V100: 20k human, 15k mouse, 10k zebrafish/chicken
   - T4: 15k human, 10k mouse, 5k zebrafish/chicken

## Troubleshooting

### Connection Errors
If you see frequent connection errors, increase `--max_retries` to 5 or more.

### Rate Limiting (429 errors)
If you hit rate limits, reduce `--chunk_size` to 10 and `--max_workers` to 4.

### Slow Downloads
If downloads are too slow, you can carefully increase `--max_workers` to 12-16, but be respectful to Ensembl's servers.

### Sequences with 'N'
Sequences containing 'N' nucleotides are automatically filtered out.

### Length Standardization
- Sequences longer than `target_length` are trimmed from the end
- Sequences shorter than `target_length` are padded with 'A' nucleotides

## Example Workflow

Complete workflow for downloading sequences for all species:

```bash
# Create output directory
mkdir -p data/processed

# Download human sequences (estimated ~33 minutes)
python scripts/download_sequences.py \
  --species human \
  --bed data/raw/GRCh38-cCREs.bed \
  --output data/processed/human_sequences.jsonl \
  --max_sequences 30000

# Download mouse sequences (estimated ~22 minutes)
python scripts/download_sequences.py \
  --species mouse \
  --bed data/raw/mm10-cCREs.bed \
  --output data/processed/mouse_sequences.jsonl \
  --max_sequences 20000

# Download zebrafish sequences (estimated ~17 minutes)
python scripts/download_sequences.py \
  --species zebrafish \
  --bed data/raw/zebrafish-enhancers.bed \
  --output data/processed/zebrafish_sequences.jsonl \
  --assembly danRer10 \
  --max_sequences 15000

# Download chicken sequences (estimated ~17 minutes)
python scripts/download_sequences.py \
  --species chicken \
  --bed data/raw/chicken-enhancers.bed \
  --output data/processed/chicken_sequences.jsonl \
  --assembly galGal4 \
  --max_sequences 15000
```

Total estimated time: ~90 minutes for all species

## Upload to Google Colab

After downloading sequences locally:

1. Upload the JSONL files to your Google Drive:
   - Place them in `/content/drive/MyDrive/enhancer_repair_research/data/processed/`
2. Run the notebook - it will automatically detect and use the cached sequences
3. Enjoy faster notebook execution without re-downloading!
