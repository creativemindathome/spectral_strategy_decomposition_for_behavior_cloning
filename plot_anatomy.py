#!/usr/bin/env python3
"""
plot_anatomy.py  —  Anatomy of a BC Residual (RH20T / v11)
===========================================================
Reproduces the 2×3 "anatomy" figure showing:
  Top-left:   Representative episode residual decomposition (raw, LF, HF)
  Top-right:  ICC bar chart (Raw, LF, HF)
  Bottom row: Scatter of residual magnitude vs action speed (3 panels)

Writes: results/fig0_anatomy.png
Usage:  python plot_anatomy.py [--output-dir results/]
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from datasets import load_dataset

warnings.filterwarnings("ignore")

FS = 10        # RH20T sampling rate
CUTOFF_HZ = 0.05   # ICC-optimal for RH20T


def load_rh20t():
    print("Loading hainh22/rh20t …")
    ds  = load_dataset("hainh22/rh20t", split="train")
    raw = ds.to_pandas()
    obs = np.stack(raw["observation.state"].values).astype(np.float32)
    act = np.stack(raw["observation.action"].values).astype(np.float32)
    ep  = raw["episode_index"].values.astype(np.int32)
    fr  = raw["frame_index"].values
    order = np.lexsort((fr, ep))
    obs, act, ep = obs[order], act[order], ep[order]
    print(f"  {len(np.unique(ep))} episodes | {len(ep)} timesteps | "
          f"obs={obs.shape[1]}D | act={act.shape[1]}D")
    return obs, act, ep


def compute_icc(scores: np.ndarray, episode_ids: np.ndarray) -> float:
    eps   = np.unique(episode_ids)
    grand = scores.mean()
    n_g, n = len(eps), len(scores)
    sizes  = np.array([np.sum(episode_ids == e) for e in eps])
    means  = np.array([scores[episode_ids == e].mean() for e in eps])
    ms_bet = np.sum(sizes * (means - grand) ** 2) / max(n_g - 1, 1)
    ss_w   = sum(((scores[episode_ids == e] - means[i]) ** 2).sum()
                 for i, e in enumerate(eps))
    ms_w   = ss_w / max(n - n_g, 1)
    k0     = (n - np.sum(sizes ** 2) / n) / max(n_g - 1, 1)
    icc    = (ms_bet - ms_w) / (ms_bet + (k0 - 1) * ms_w + 1e-12)
    return float(max(0.0, icc))


def cross_fit_residuals(obs, act, ep):
    """GroupKFold Ridge on all episodes (no leakage concern for this figure)."""
    ridge = Ridge(alpha=1.0)
    gkf   = GroupKFold(n_splits=5)
    pred  = np.zeros_like(act)
    for tr_i, te_i in gkf.split(obs, act, groups=ep):
        ridge.fit(obs[tr_i], act[tr_i])
        pred[te_i] = ridge.predict(obs[te_i])
    return act - pred


def decompose_lf_hf_butterworth(resid, ep_ids, cutoff_hz):
    """Butterworth lowpass — used for ICC computation (spectral test)."""
    nyq  = FS / 2.0
    norm = np.clip(cutoff_hz / nyq, 1e-4, 0.9999)
    b, a = butter(4, norm, btype="low")
    lf   = np.zeros_like(resid)
    min_len = 4 * 4 + 1
    for ep in np.unique(ep_ids):
        mask = ep_ids == ep
        r_ep = resid[mask]
        if len(r_ep) < min_len:
            lf[mask] = r_ep.mean(axis=0)
        else:
            for d in range(resid.shape[1]):
                lf[mask, d] = filtfilt(b, a, r_ep[:, d])
    hf = resid - lf
    return lf, hf


def decompose_lf_hf_ridge(resid, ep_ids, poly_degree=4):
    """Ridge polynomial trend fit — smoother visual for short episodes.

    Fits a degree-`poly_degree` polynomial in time per episode to extract
    the slow trend (LF). HF = residual - trend. Better than Butterworth for
    episodes shorter than ~2 LF cycles.
    """
    from sklearn.preprocessing import PolynomialFeatures
    lf = np.zeros_like(resid)
    for ep in np.unique(ep_ids):
        mask  = ep_ids == ep
        n_ep  = mask.sum()
        t     = np.linspace(0, 1, n_ep).reshape(-1, 1)
        poly  = PolynomialFeatures(degree=poly_degree, include_bias=True)
        T     = poly.fit_transform(t)
        for d in range(resid.shape[1]):
            r_ep      = resid[mask, d]
            reg       = Ridge(alpha=1e-3).fit(T, r_ep)
            lf[mask, d] = reg.predict(T)
    hf = resid - lf
    return lf, hf


def action_speed(act, ep_ids):
    speed = np.zeros(len(act))
    for ep in np.unique(ep_ids):
        mask = np.where(ep_ids == ep)[0]
        diff = np.diff(act[mask], axis=0)
        s    = np.linalg.norm(diff, axis=1)
        speed[mask[0]] = 0.0
        speed[mask[1:]] = s
    return speed


def pick_representative_episode(resid, ep_ids):
    """Pick episode in the upper quartile of RMS — visually dynamic but not an outlier."""
    eps = np.unique(ep_ids)
    rms = np.array([np.sqrt(np.mean(resid[ep_ids == e] ** 2)) for e in eps])
    q75 = np.percentile(rms, 75)
    # closest to 75th percentile → dynamic but not the single most extreme
    return eps[np.argmin(np.abs(rms - q75))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results")
    p.add_argument("--cutoff-hz", type=float, default=CUTOFF_HZ)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    obs, act, ep = load_rh20t()
    resid        = cross_fit_residuals(obs, act, ep)

    # Butterworth decomposition — used for ICC and scatter (matches run_v11.py)
    lf_bw, hf_bw = decompose_lf_hf_butterworth(resid, ep, args.cutoff_hz)

    # Ridge polynomial decomposition — used for the episode trace only (cleaner visual)
    lf_ridge, hf_ridge = decompose_lf_hf_ridge(resid, ep, poly_degree=4)

    # ── ICC (from Butterworth — consistent with run_v11.py) ───────────────────
    mag_raw = np.linalg.norm(resid,  axis=1)
    mag_lf  = np.linalg.norm(lf_bw, axis=1)
    mag_hf  = np.linalg.norm(hf_bw, axis=1)
    icc_raw = compute_icc(mag_raw, ep)
    icc_lf  = compute_icc(mag_lf,  ep)
    icc_hf  = compute_icc(mag_hf,  ep)
    print(f"  ICC — raw={icc_raw:.3f}  LF={icc_lf:.3f}  HF={icc_hf:.3f}")

    # ── Speed correlation (from Butterworth) ─────────────────────────────────
    speed = action_speed(act, ep)
    from scipy.stats import spearmanr
    rho_raw, _ = spearmanr(mag_raw, speed)
    rho_lf,  _ = spearmanr(mag_lf,  speed)
    rho_hf,  _ = spearmanr(mag_hf,  speed)
    print(f"  ρ(speed) — raw={rho_raw:.3f}  LF={rho_lf:.3f}  HF={rho_hf:.3f}")

    # ── Representative episode ────────────────────────────────────────────────
    rep_ep   = pick_representative_episode(resid, ep)
    mask_rep = ep == rep_ep
    t_rep    = np.arange(mask_rep.sum()) / FS   # seconds

    # Episode trace uses Ridge decomposition (smoother for short episodes)
    r_trace  = resid[mask_rep, 0]
    lf_trace = lf_ridge[mask_rep, 0]
    hf_trace = hf_ridge[mask_rep, 0]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Anatomy of a BC Residual: Spectral Decomposition Evidence (RH20T — v11)\n"
        f"ICC_LF={icc_lf:.3f}  ICC_raw={icc_raw:.3f}  ICC_HF={icc_hf:.3f}  "
        f"(Butterworth {args.cutoff_hz} Hz) | episode trace: Ridge poly-4 trend",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35,
                          left=0.07, right=0.97, top=0.91, bottom=0.08)

    # ── Top-left: episode decomposition trace ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(t_rep, r_trace,  color="#9CA3AF", alpha=0.9,  lw=1.2, label="Raw residual")
    ax0.plot(t_rep, lf_trace, color="#2563EB", lw=2.2,     label="LF trend (Ridge poly-4)")
    ax0.plot(t_rep, hf_trace, color="#DC2626", alpha=0.75, lw=1.0, label="HF residual (raw − trend)")
    ax0.axhline(0, color="black", lw=0.7, ls="--")
    ax0.set_xlabel("Time (s)", fontsize=11)
    ax0.set_ylabel("Residual (dim 0)", fontsize=11)
    ax0.set_title(f"Representative Episode (ep {rep_ep}): Residual Decomposition", fontsize=12)
    ax0.legend(fontsize=9, loc="upper right")
    ax0.tick_params(labelsize=9)

    # ── Top-right: ICC bar chart ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 2])
    bar_vals   = [icc_raw, icc_lf, icc_hf]
    bar_labels = ["Raw", "LF", "HF"]
    bar_colors = ["#9CA3AF", "#2563EB", "#F87171"]
    bars = ax1.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, bar_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("ICC (between-episode / total)", fontsize=10)
    ax1.set_title("Episode-level Structure (ICC)\nHigher = more episode-level variation", fontsize=11)
    ax1.set_ylim(0, max(bar_vals) * 1.25 + 0.02)
    ax1.tick_params(labelsize=10)

    # ── Bottom row: 3 scatter panels ──────────────────────────────────────────
    scatter_cfg = [
        (mag_raw, "Raw residual magnitude",  "Raw residual",  "#9CA3AF", rho_raw),
        (mag_lf,  "LF component magnitude",  "LF component",  "#2563EB", rho_lf),
        (mag_hf,  "HF component magnitude",  "HF component",  "#F87171", rho_hf),
    ]

    # Subsample for rendering speed (≤8k points)
    rng   = np.random.default_rng(0)
    n_pts = min(8000, len(speed))
    idx   = rng.choice(len(speed), n_pts, replace=False)

    for col, (mag, xlabel, title, color, rho) in enumerate(scatter_cfg):
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(mag[idx], speed[idx], color=color, alpha=0.25, s=5, linewidths=0)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Action speed", fontsize=10)
        ax.set_title(f"{title}\nrho(speed) = {rho:.3f}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

    out_path = os.path.join(args.output_dir, "fig0_anatomy.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
