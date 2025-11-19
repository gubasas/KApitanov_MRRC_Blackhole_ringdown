# Kapitanov Horizon Quantization — Detection pipeline

This repository contains a minimal pipeline to fetch LIGO strain data, extract ringdown windows, compute FFT/PSD, and search for discrete-frequency combs as described in the Kapitanov quantization proposal.

Quickstart

1. Create a virtualenv and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the pipeline for a single event (example):

```bash
python download_gwpy.py --event GW200129_065458 --ringdown-start 0.01 --ringdown-duration 0.09
```

The script will download open LIGO data for the requested event, extract the ringdown and background windows, compute PSDs, and report a simple detection statistic for a provided comb hypothesis.

Notes
- The code is a starting point and is intentionally minimal. It is designed for iterative improvements: better noise estimation, more sophisticated stacking, and statistical tests should be added before publication.
- See `download_gwpy.py` docstrings for details on parameters and usage.
