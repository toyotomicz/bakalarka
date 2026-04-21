# Lossless Image Compression Benchmark

A benchmarking tool for comparing lossless image compression algorithms. Measures compression ratio, speed, and system resource usage across multiple formats and datasets.

## Supported algorithms

| Algorithm | Format | Interface |
|-----------|--------|-----------|
| CharLS | JPEG-LS | DLL (C++) |
| libpng | PNG | DLL (C) |
| OptiPNG | PNG | CLI |
| OxiPNG | PNG | CLI |
| Pillow PNG / WebP / TIFF | PNG, WebP, TIFF | Python API |
| QOI | QOI | Python API |
| WebP Lossless (cwebp) | WebP | CLI |

## Requirements

- Windows 64-bit
- Python 3.10+

Native binaries (OptiPNG, OxiPNG, cwebp, dwebp, CharLS DLL, libpng DLL) are bundled in `src/libs/` and require no separate installation.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src
python main.py
```

The GUI lets you select input images, choose algorithms and compression levels (FASTEST / BALANCED / BEST), configure iteration count, and optionally enable CPU isolation and resource monitoring. Results are saved as JSON to `benchmark_results/json_reports/`.

## Running tests

```bash
python -m pytest --cov
```

Test coverage: [![codecov](https://codecov.io/gh/toyotomicz/bakalarka/graph/badge.svg?token=0f4458ce-b624-4ee6-b429-c697c692f599)](https://codecov.io/gh/toyotomicz/bakalarka)
