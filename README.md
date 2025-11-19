# Kapitanov Horizon Quantization — Detection pipeline

This repository contains a minimal, reproducible pipeline to search for discrete-frequency "comb" structure in black-hole ringdown data using LIGO open strain data.

This project is experimental research code intended to reproduce the PHASE-2 (exact-mode) analysis described in the roadmap. Use responsibly and verify results before drawing scientific conclusions.

## Contents
- `download_gwpy.py` — main CLI and analysis functions (data fetch, PSD, detection, bootstrap, stacking, masking)
- `requirements.txt` — Python dependencies
- `results/` — local directory where per-event JSON outputs, bootstraps and plots are written (by default this folder is git-ignored; a small set of sample results are included in the repo for demonstration)

## Quickstart

1. Create and activate a Python virtual environment, then install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. To run the script with a sample result (no network required), simply run the script with no arguments — it will print a short summary of the included sample result (if `results/*_result.json` exists):

```bash
python download_gwpy.py
```

3. Run a single-event exact-mode analysis (PHASE 2: 10–100 ms ringdown, 1–5 s background, Tukey α=0.25):

```bash
python download_gwpy.py --event GW200129_065458 --detector H1 --exact
```

4. Run the validation suite (detector confirmation, time-window robustness, k-scan, pre-merger):

```bash
python download_gwpy.py --event GW200129_065458 --validate
```

5. Run per-event bootstrap for a list of events (empirical p-values). Example uses k=2 and 2000 iterations:

```bash
python download_gwpy.py --do-bootstrap --events GW200129_065458,GW200224_222234 --kvalue 2 --bootstrap-iters 2000
```

6. Mask known instrumental-line harmonics (±5 Hz) during detection/bootstrapping:

```bash
python download_gwpy.py --do-bootstrap --events GW200129_065458 --kvalue 2 --bootstrap-iters 2000 --mask-lines --mask-width 5.0
```

7. Stack mass-normalized PSDs across events and bootstrap the stacked statistic:

```bash
python download_gwpy.py --do-stack --events GW200129_065458,GW200224_222234 --kvalue 2 --stack-iters 1000
```

## Outputs
- Per-event JSON results: `results/{event}_{detector}_result.json`
- Bootstrap outputs: `results/{event}_{detector}_bootstrap_k{k}.json`
- Stacking outputs: `results/stacking_k{k}.json`
- CSV summary: `results/k2_summary.csv`
- Diagnostic plots: `results/plots/*.png`

Note: by default `results/` is included in `.gitignore`. A curated set of sample results and one diagnostic plot are pushed to the repository so users can run the script without network access and inspect example outputs.

## Interpreting results
- `detection_stat` — the comb detection statistic (sum of harmonic SNRs normalized by sqrt(N)). Larger values indicate a stronger comb-like signal.
- `harmonic_snrs` — per-harmonic SNR values (units: power ratio to median background PSD in the harmonic band).
- `bootstrap_stats` — the empirical null distribution obtained by resampling background segments. The reported `pvalue` is the fraction of bootstrap statistics ≥ `real_stat`.

Statistical caution: empirical p-values depend on the chosen background interval and bootstrap implementation. Use masking, multiple time-window checks, and physical cross-checks before drawing conclusions.

## Troubleshooting
- If `gwpy`/`gwosc` fail to fetch data (network or GWOSC availability), re-run when network is available or run the script with no arguments to inspect sample outputs included in the repository.
- If an event's detector data is missing from GWOSC for the requested window, the code will log an error for that detector and skip or fall back as appropriate.

## Development & contributions
- See `CONTRIBUTING.md` for guidelines.
- License: MIT (`LICENSE`)

## Contact
- Open an issue or PR on the GitHub repository for questions or improvements.

---
Minimal, reproducible research code — use responsibly.

## Citations

This implementation was developed as an exploratory pipeline following the Kapitanov horizon-quantization proposal. If you use this code or the results derived from it, please cite the original proposal by Fedor Kapitanov:

- Kapitanov, Fedor. "Quantization of horizon frequencies of black holes and observational signatures." viXra: AI (2025). PDF: https://ai.vixra.org/pdf/2511.0009v1.pdf

Suggested BibTeX:

```bibtex
@article{kapitanov2025quantization,
	title = {Quantization of horizon frequencies of black holes and observational signatures},
	author = {Kapitanov, Fedor},
	year = {2025},
	note = {viXra: AI. \url{https://ai.vixra.org/pdf/2511.0009v1.pdf}}
}
```

If you adapt or extend this code for publication, please include a citation to the repository as well (e.g., the GitHub URL and commit hash used in your analysis).
