# NFLDataAnalysis

Repository for NFL data analysis projects.

## RPO analysis script

This repository now includes a standalone script to analyze Run-Pass Option (RPO) usage and efficiency from Pro Football Reference advanced passing data.

### Install dependencies

```bash
pip install nflreadpy pandas numpy
```

### Run analysis

```bash
python rpo_analysis.py --start-season 2022 --end-season 2025 --top-n 10
```

Optional: save CSV outputs.

```bash
python rpo_analysis.py --start-season 2022 --end-season 2025 --output-dir /tmp/rpo_outputs
```
