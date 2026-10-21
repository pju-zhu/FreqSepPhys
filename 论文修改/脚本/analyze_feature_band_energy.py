"""Analyze exported real features; no model loading or synthetic experiment results.
Manifest JSON: list of {path, dataset, subject, clip, stream, layer}.
Each NPZ: input, smooth, residual (T,C,H,W or T,D), fs (scalar Hz).
Export BEFORE nonlinear encoding. fs must describe feature sampling, not video fps.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def energy(x, fs):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or x.shape[0] < 8 or not np.isfinite(x).all():
        raise ValueError('Features must be finite arrays with time first and T >= 8')
    t = x.shape[0]
    x = x.reshape(t, -1)
    dc = float(np.mean(np.mean(x, axis=0)**2))
    x = x - x.mean(axis=0, keepdims=True)
    w = np.hanning(t)
    power = np.abs(np.fft.rfft(x*w[:, None], axis=0))**2/(fs*np.sum(w*w))
    power[1:-1 if t % 2 == 0 else None] *= 2
    power = power.mean(axis=1)  # Average powers, not features (avoid cancellation).
    f = np.fft.rfftfreq(t, 1/fs)
    df = fs/t
    masks = [f < .5, (f >= .5) & (f <= 3), f > 3]
    values = [float(power[m].sum()*df) for m in masks]
    total = sum(values)
    return values, [v/total if total > 0 else None for v in values], dc, df, [int(m.sum()) for m in masks]


def run(manifest, output):
    records = json.loads(manifest.read_text())
    if not records:
        raise ValueError('Empty manifest')
    rows = []
    keys = set()
    for rec in records:
        identity = tuple(str(rec[k]) for k in ('dataset','subject','clip','stream','layer'))
        if identity in keys:
            raise ValueError('Duplicate sample identity: '+str(identity))
        keys.add(identity)
        with np.load(manifest.parent/rec['path'], allow_pickle=False) as z:
            fs = float(z['fs'])
            if not np.isfinite(fs) or fs <= 6:
                raise ValueError('fs must exceed 6 Hz to analyze >3 Hz band')
            arrays = [z[k] for k in ('input','smooth','residual')]
            if any(a.shape != arrays[0].shape for a in arrays):
                raise ValueError('Input and decomposition arrays must have identical shapes')
            if not np.allclose(arrays[0], arrays[1]+arrays[2], rtol=1e-4, atol=1e-6):
                raise ValueError('Export immediately around decomposition: input != smooth + residual')
            base = energy(arrays[0], fs)[0]
            for stage, x in zip(('input','smooth','residual'), arrays):
                vals, shares, dc, df, counts = energy(x, fs)
                for i, band in enumerate(('below_0.5','0.5_to_3','above_3')):
                    rows.append(dict(zip(('dataset','subject','clip','stream','layer'),identity),
                        stage=stage,band=band,fs_Hz=fs,nyquist_Hz=fs/2,T=x.shape[0],df_Hz=df,
                        bins=counts[i],energy=vals[i],share=shares[i],
                        output_input_band_ratio=vals[i]/base[i] if base[i]>0 else None,
                        dc_mean_square=dc))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'{len(records)} clips; {len(rows)} rows -> {output}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.manifest,a.output)
