#!/usr/bin/env python3
"""Compute CL and Gaussian sigma equivalents for summary CSV.

Reads `results/k2_summary.csv` and writes `results/k2_summary_with_significance.csv`.
Columns added: `CL_percent`, `sigma_one_sided`, `sigma_two_sided`.
"""
import csv
import math
from pathlib import Path
from scipy.stats import norm


def safe_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == '' or s.upper() == 'N/A':
            return None
        return float(s)
    except Exception:
        return None


def formatf(x, prec=4):
    if x is None:
        return ''
    return f"{x:.{prec}f}"


def main():
    base = Path('results')
    infile = base / 'k2_summary.csv'
    outfile = base / 'k2_summary_with_significance.csv'
    if not infile.exists():
        print('Input CSV not found:', infile)
        return

    with infile.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    out_rows = []
    for r in rows:
        p = safe_float(r.get('p_value'))
        cl = None
        sigma_one = None
        sigma_two = None
        if p is not None:
            # CL as percentage
            cl = (1.0 - p) * 100.0
            # one-sided sigma: z such that P(Z>z)=p
            try:
                if p <= 0:
                    sigma_one = float('inf')
                elif p >= 1:
                    sigma_one = 0.0
                else:
                    sigma_one = float(norm.isf(p))
                # two-sided sigma: use p/2 in tail
                p2 = min(max(p / 2.0, 1e-323), 1.0 - 1e-16)
                sigma_two = float(norm.isf(p2))
            except Exception:
                sigma_one = None
                sigma_two = None

        r2 = dict(r)
        r2['CL_percent'] = formatf(cl, 3) if cl is not None else ''
        r2['sigma_one_sided'] = formatf(sigma_one, 3) if sigma_one is not None and math.isfinite(sigma_one) else ''
        r2['sigma_two_sided'] = formatf(sigma_two, 3) if sigma_two is not None and math.isfinite(sigma_two) else ''
        out_rows.append(r2)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    # ensure new columns at the end
    for col in ('CL_percent', 'sigma_one_sided', 'sigma_two_sided'):
        if col not in fieldnames:
            fieldnames.append(col)

    with outfile.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print('Wrote', outfile)


if __name__ == '__main__':
    main()
