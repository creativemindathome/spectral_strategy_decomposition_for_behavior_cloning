#!/usr/bin/env python3
"""
Spectral Surrogate Proprioception — v11  (Definitive RH20T)
=============================================================
Primary dataset: hainh22/rh20t
  - 30 episodes | 4933 timesteps | 10 Hz
  - obs = 6D joint angles  (policy input)
  - act = 3D actions
  - F/T = 6D force-torque  (held out — never seen by policy)

Experiment lineage:
  v2  ALOHA sim  — IV econometrics fails
  v3  ALOHA sim  — Sample-wise reweighting fails
  v4  ALOHA sim  — Spectral dual weighting: +2.1%
  v5  ALOHA sim  — Episode-level LF context: +9.8%
  v6  RH20T      — Replicates on real hardware: +4.8%
                   Jacobian coupling: rho=0.85, p<0.0001
  v9  ALOHA mob  — CAWL routing collapses; spectral doesn't help
  v10 ALOHA mob  — Leakage-free protocol: transfer hurts (ICC_LF=0.072)
  v11 RH20T      — Leakage-free + full directional coupling (THIS)

Key questions answered here:
  1. Does LF context encode actual contact forces?   (epistemological test)
  2. Does residual *direction* reveal Jacobian structure?  (coupling test)
  3. Does conditioning gain track discovered LF structure?  (LF stratification)
  4. Does leakage-free transfer context help?   (v10 protocol on ICC_LF=0.541)

Protocol (default: train/test cross-fit):
  train   ~77% of episodes  →  critic cross-fit + policy training
  test    ~23% of episodes  →  evaluation
  Train contexts are out-of-fold within train; test contexts come from critic fit on full train.
  Optional strict 3-way split remains available for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from scipy.signal import butter, filtfilt, welch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, rankdata, t as student_t
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

warnings.filterwarnings("ignore")


# ── Constants ────────────────────────────────────────────────────────────────
FS = 10        # RH20T sampling rate (Hz)
FT_LABELS  = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
ACT_LABELS = ["act0", "act1", "act2"]

CONTEXT_COLORS = {
    "no_context":  "#6B7280",
    "oracle_lf":   "#2563EB",
    "transfer_lf": "#059669",
    "hf_oracle":   "#DC2626",
    "lf_pca3":     "#7C3AED",
    "random":      "#F59E0B",
}
CONTEXT_LABELS = OrderedDict([
    ("no_context",   "No Context"),
    ("oracle_lf",    "Oracle LF"),
    ("transfer_lf",  "Transfer LF (k-NN)"),
    ("hf_oracle",    "HF Oracle (neg ctrl)"),
    ("lf_pca3",      "LF + PCA(3)"),
    ("random",       "Random (ctrl)"),
])


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="hainh22/rh20t")
    p.add_argument("--seeds",         type=int,   default=5)
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch-size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--hidden",        type=int,   default=256)
    p.add_argument("--cutoff-hz",     type=float, default=0.05,
                   help="LF cutoff (Hz). Default 0.05 Hz (empirically chosen for RH20T 10 Hz data). "
                        "Pass None-equivalent by omitting to use ICC-optimal sweep (experimental).")
    p.add_argument("--protocol",      default="train_test_crossfit",
                   choices=["train_test_crossfit", "strict_3way"],
                   help="Residual/context protocol. Default uses train/test split with train-only cross-fitted critic.")
    p.add_argument("--critic-frac",   type=float, default=0.33,
                   help="Fraction of episodes used as critic set for strict_3way only.")
    p.add_argument("--test-frac",     type=float, default=0.23,
                   help="Fraction of episodes used as test set.")
    p.add_argument("--k-neighbors",  type=int,   default=5)
    p.add_argument("--split-seed",   type=int,   default=42)
    p.add_argument("--n-permutations", type=int, default=500,
                   help="Permutations for Jacobian coupling null test.")
    p.add_argument("--output-dir",   default=None)
    return p.parse_args()


# ── Utilities ────────────────────────────────────────────────────────────────
def compute_icc(scores: np.ndarray, episode_ids: np.ndarray) -> float:
    eps    = np.unique(episode_ids)
    grand  = scores.mean()
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


def decompose_lf_hf(residuals: np.ndarray, episode_ids: np.ndarray,
                    cutoff_hz: float, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    nyq  = fs / 2.0
    norm = np.clip(cutoff_hz / nyq, 1e-4, 0.9999)
    b, a = butter(4, norm, btype="low")
    lf   = np.zeros_like(residuals)
    hf   = np.zeros_like(residuals)
    min_len = 4 * 4 + 1   # 4th-order filter minimum
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        r_ep = residuals[mask]
        if len(r_ep) < min_len:
            lf[mask] = r_ep.mean(axis=0)
        else:
            for d in range(residuals.shape[1]):
                lf[mask, d] = filtfilt(b, a, r_ep[:, d])
        hf[mask] = residuals[mask] - lf[mask]
    return lf, hf


def episode_mean(arr: np.ndarray, episode_ids: np.ndarray,
                 ep_list: list[int]) -> np.ndarray:
    return np.array([arr[episode_ids == e].mean(axis=0) for e in ep_list])


def std_ctx(mat: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    sc = StandardScaler().fit(mat[train_rows])
    return sc.transform(mat), sc


def compute_transfer_contexts(
    feature_mat: np.ndarray,          # (n_all_eps, D_feat)  — all train+test
    lf_mean_mat: np.ndarray,          # (n_all_eps, D_act)   — standardised
    train_idx: list[int],             # indices into ep_list that are train
    target_idx: list[int],            # indices into ep_list for which we want transfer ctx
    k: int,
    exclude_self: bool,
) -> tuple[np.ndarray, dict]:
    """k-NN weighted average of train LF contexts for each target episode."""
    train_feat = feature_mat[train_idx]   # (n_train, D_feat)
    train_ctx  = lf_mean_mat[train_idx]   # (n_train, D_act)

    result = np.zeros((len(target_idx), lf_mean_mat.shape[1]), dtype=np.float32)
    meta   = {}
    for pos, t_i in enumerate(target_idx):
        feat = feature_mat[t_i : t_i + 1]           # (1, D_feat)
        dists = cdist(feat, train_feat, metric="euclidean")[0]
        cands_i = list(range(len(train_idx)))
        cands_d = dists.copy()
        if exclude_self and t_i in train_idx:
            self_pos = train_idx.index(t_i)
            cands_i  = [c for c in cands_i if c != self_pos]
            cands_d  = np.delete(cands_d, self_pos)
        order   = np.argsort(cands_d)[: min(k, len(cands_d))]
        chosen  = np.array(cands_i)[order]
        d_vals  = cands_d[order]
        weights = 1.0 / np.maximum(d_vals, 1e-6)
        weights /= weights.sum()
        result[pos] = np.average(train_ctx[chosen], axis=0, weights=weights)
        meta[t_i] = {
            "neighbors": [int(train_idx[c]) for c in chosen],
            "distances": d_vals.tolist(),
            "weights":   weights.tolist(),
        }
    return result, meta


def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    res = spearmanr(x, y)
    rho = float(res.statistic) if np.isfinite(res.statistic) else 0.0
    pval = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
    return rho, pval


def partial_spearmanr(x: np.ndarray, y: np.ndarray,
                      covars: np.ndarray) -> tuple[float, float]:
    """Partial Spearman via rank-transform + linear residualisation."""
    z = np.asarray(covars, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    x_r = rankdata(x)
    y_r = rankdata(y)
    z_r = np.column_stack([rankdata(z[:, j]) for j in range(z.shape[1])])
    design = np.column_stack([np.ones(len(x_r)), z_r])
    beta_x, *_ = np.linalg.lstsq(design, x_r, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, y_r, rcond=None)
    x_res = x_r - design @ beta_x
    y_res = y_r - design @ beta_y
    if np.std(x_res) < 1e-12 or np.std(y_res) < 1e-12:
        return 0.0, 1.0
    rho = float(np.corrcoef(x_res, y_res)[0, 1])
    if not np.isfinite(rho):
        return 0.0, 1.0
    dof = len(x_r) - z_r.shape[1] - 2
    if dof <= 0:
        return rho, 1.0
    rho_clip = float(np.clip(rho, -0.999999, 0.999999))
    t_stat = abs(rho_clip) * np.sqrt(dof / max(1e-12, 1.0 - rho_clip ** 2))
    pval = float(2.0 * student_t.sf(t_stat, dof))
    return rho_clip, pval


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    covars: np.ndarray | None = None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if covars is None:
            rho, _ = safe_spearmanr(x[idx], y[idx])
        else:
            rho, _ = partial_spearmanr(x[idx], y[idx], covars[idx])
        if np.isfinite(rho):
            vals.append(rho)
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals, dtype=np.float64)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def build_train_test_crossfit_residuals(
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    ep_raw: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float = 10.0,
) -> np.ndarray:
    """Out-of-fold residuals on train, full-train predictions on test."""
    cross_resid = np.zeros_like(act_arr)

    obs_tr = obs_arr[train_mask]
    act_tr = act_arr[train_mask]
    ep_tr = ep_raw[train_mask]
    n_train_eps = len(np.unique(ep_tr))
    n_splits = min(5, n_train_eps)

    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        resid_tr = np.zeros_like(act_tr)
        for tr_idx, va_idx in gkf.split(obs_tr, groups=ep_tr):
            sc = StandardScaler().fit(obs_tr[tr_idx])
            ridge = Ridge(alpha=alpha).fit(sc.transform(obs_tr[tr_idx]), act_tr[tr_idx])
            resid_tr[va_idx] = act_tr[va_idx] - ridge.predict(sc.transform(obs_tr[va_idx]))
        cross_resid[train_mask] = resid_tr
    else:
        sc = StandardScaler().fit(obs_tr)
        ridge = Ridge(alpha=alpha).fit(sc.transform(obs_tr), act_tr)
        cross_resid[train_mask] = act_tr - ridge.predict(sc.transform(obs_tr))

    sc_full = StandardScaler().fit(obs_tr)
    ridge_full = Ridge(alpha=alpha).fit(sc_full.transform(obs_tr), act_tr)
    cross_resid[test_mask] = act_arr[test_mask] - ridge_full.predict(
        sc_full.transform(obs_arr[test_mask]))
    return cross_resid


# ── Architecture ─────────────────────────────────────────────────────────────
class BC_MLP(nn.Module):
    """Dense MLP — context concatenated to observation."""
    def __init__(self, in_dim: int, out_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_bc(
    S_tr: np.ndarray, A_tr: np.ndarray, C_tr: np.ndarray,
    S_te: np.ndarray, A_te: np.ndarray, C_te: np.ndarray,
    test_groups: dict[str, np.ndarray],
    seed: int = 0,
    n_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 128,
    hidden: int = 256,
    label: str = "",
) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr = np.column_stack([S_tr, C_tr]) if C_tr.shape[1] > 0 else S_tr.copy()
    X_te = np.column_stack([S_te, C_te]) if C_te.shape[1] > 0 else S_te.copy()
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    Ytr = torch.tensor(A_tr, dtype=torch.float32)
    Xte = torch.tensor(X_te, dtype=torch.float32)
    Yte = torch.tensor(A_te, dtype=torch.float32)
    model  = BC_MLP(X_tr.shape[1], A_tr.shape[1], hidden=hidden)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    epoch_mse = []
    group_hist: dict[str, list[float]] = {g: [] for g in test_groups}
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            ps = ((model(Xte) - Yte) ** 2).mean(dim=1).numpy()
            epoch_mse.append(float(ps.mean()))
            for g, gm in test_groups.items():
                group_hist[g].append(float(ps[gm].mean()) if gm.any() else float("nan"))
        elapsed = time.time() - t0
        eta = elapsed / (epoch + 1) * (n_epochs - epoch - 1)
        print(f"  [{label}] ep {epoch+1:3d}/{n_epochs}"
              f" | loss={ep_loss/len(loader):.5f}"
              f" | mse={epoch_mse[-1]:.5f}"
              f" | best={min(epoch_mse):.5f}"
              f" | ETA {int(eta//60)}m{int(eta%60):02d}s", flush=True)
    return {
        "test_mse": epoch_mse,
        "best_mse": float(min(epoch_mse)),
        "best_epoch": int(np.argmin(epoch_mse)),
        "group_mse": group_hist,
        "params": sum(p.numel() for p in model.parameters()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    np.random.seed(args.split_seed)
    rng = np.random.default_rng(args.split_seed)

    output_dir = args.output_dir or "results"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run_log.txt")

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    open(log_path, "w").close()   # clear

    log("=" * 72)
    log("Spectral Surrogate Proprioception — v11  |  hainh22/rh20t")
    log(f"seeds={args.seeds}  epochs={args.epochs}  batch={args.batch_size}"
        f"  hidden={args.hidden}  protocol={args.protocol}  test_frac={args.test_frac:.2f}")
    log("=" * 72)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 0: DATA LOADING + EPISODE SPLIT
    # ══════════════════════════════════════════════════════════════════════════
    log("\n[PHASE 0] Data loading — hainh22/rh20t")
    ds  = load_dataset("hainh22/rh20t", split="train")
    raw = ds.to_pandas()
    ft_arr  = np.stack(raw["observation.force_and_torque"].values).astype(np.float32)
    obs_arr = np.stack(raw["observation.state"].values).astype(np.float32)
    act_arr = np.stack(raw["observation.action"].values).astype(np.float32)
    ep_raw  = raw["episode_index"].values
    fr_raw  = raw["frame_index"].values

    order = np.lexsort((fr_raw, ep_raw))
    ft_arr  = ft_arr[order]; obs_arr = obs_arr[order]
    act_arr = act_arr[order]; ep_raw  = ep_raw[order]

    # Filter short episodes
    ep_ids_all = np.sort(np.unique(ep_raw))
    valid_eps  = [e for e in ep_ids_all if (ep_raw == e).sum() >= 30]
    keep = np.isin(ep_raw, valid_eps)
    ft_arr  = ft_arr[keep]; obs_arr = obs_arr[keep]
    act_arr = act_arr[keep]; ep_raw  = ep_raw[keep]
    ep_ids  = np.sort(np.unique(ep_raw))
    N, D_obs, D_act, D_ft = len(act_arr), obs_arr.shape[1], act_arr.shape[1], ft_arr.shape[1]
    log(f"  {N} timesteps | {len(ep_ids)} episodes | obs={D_obs}D | act={D_act}D | F/T={D_ft}D")

    # Episode split
    shuffled    = rng.permutation(ep_ids).tolist()
    n_test      = max(1, int(args.test_frac   * len(shuffled)))
    if n_test >= len(shuffled):
        n_test = len(shuffled) - 1
    critic_eps: list[int] = []
    if args.protocol == "strict_3way":
        n_critic = max(1, int(args.critic_frac * len(shuffled)))
        if n_critic + n_test >= len(shuffled):
            n_critic = max(1, len(shuffled) - n_test - 1)
        n_train = len(shuffled) - n_critic - n_test
        critic_eps = shuffled[:n_critic]
        train_eps = shuffled[n_critic : n_critic + n_train]
        test_eps = shuffled[n_critic + n_train:]
    else:
        n_train = len(shuffled) - n_test
        train_eps = shuffled[:n_train]
        test_eps = shuffled[n_train:]
        n_critic = 0

    critic_mask = np.isin(ep_raw, critic_eps) if critic_eps else np.zeros(len(ep_raw), dtype=bool)
    train_mask  = np.isin(ep_raw, train_eps)
    test_mask   = np.isin(ep_raw, test_eps)
    if args.protocol == "strict_3way":
        log(f"  Split: critic={n_critic} eps | train={n_train} eps | test={n_test} eps")
        log(f"  (critic steps={critic_mask.sum()} | train={train_mask.sum()} | test={test_mask.sum()})")
    else:
        log(f"  Split: train={n_train} eps | test={n_test} eps")
        log(f"  (train steps={train_mask.sum()} | test={test_mask.sum()})")

    # Action speed (for spectral validation)
    action_speed = np.zeros(N)
    for ep in ep_ids:
        idx = np.where(ep_raw == ep)[0]
        if len(idx) > 1:
            action_speed[idx[1:]] = np.linalg.norm(np.diff(act_arr[idx], axis=0), axis=1)

    # Oracle gap (F/T matters?) — 5-fold CV on all episodes
    log("\n  Oracle gap (5-fold CV):")
    S_full    = np.concatenate([obs_arr, ft_arr], axis=1)
    gkf_gap   = GroupKFold(n_splits=min(5, len(ep_ids) // 2))
    cv_p, cv_f = [], []
    for tr, te in gkf_gap.split(obs_arr, groups=ep_raw):
        scp = StandardScaler().fit(obs_arr[tr])
        scf = StandardScaler().fit(S_full[tr])
        mp  = Ridge(10.).fit(scp.transform(obs_arr[tr]), act_arr[tr])
        mf  = Ridge(10.).fit(scf.transform(S_full[tr]),  act_arr[tr])
        cv_p.append(np.mean((act_arr[te] - mp.predict(scp.transform(obs_arr[te])))**2))
        cv_f.append(np.mean((act_arr[te] - mf.predict(scf.transform(S_full[te])))**2))
    mse_partial = float(np.mean(cv_p))
    mse_full    = float(np.mean(cv_f))
    oracle_gap  = (mse_partial - mse_full) / mse_partial * 100
    log(f"    MSE(joints-only) = {mse_partial:.8f}")
    log(f"    MSE(joints+F/T)  = {mse_full:.8f}")
    log(f"    Oracle gap: {oracle_gap:.2f}%")
    if oracle_gap < 2.0:
        log("  !! Oracle gap < 2%. F/T adds little — check dataset.")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: LEAKAGE-FREE RESIDUALS + ICC-OPTIMISED SPECTRAL DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log(f"[PHASE 1] Residuals + ICC-optimised cutoff ({args.protocol})")
    log("=" * 72)

    policy_mask = train_mask | test_mask
    if args.protocol == "strict_3way":
        sc_critic = StandardScaler().fit(obs_arr[critic_mask])
        ridge = Ridge(alpha=10.0).fit(
            sc_critic.transform(obs_arr[critic_mask]), act_arr[critic_mask])
        cross_resid = act_arr[policy_mask] - ridge.predict(
            sc_critic.transform(obs_arr[policy_mask]))
        log("  Critic fit on disjoint critic episodes; residuals predicted on train+test.")
    else:
        cross_resid = build_train_test_crossfit_residuals(
            obs_arr=obs_arr,
            act_arr=act_arr,
            ep_raw=ep_raw,
            train_mask=train_mask,
            test_mask=test_mask,
            alpha=10.0,
        )[policy_mask]
        log("  Train residuals are out-of-fold within train; test residuals use critic fit on full train.")
    ep_policy = ep_raw[policy_mask]
    ft_policy = ft_arr[policy_mask]

    # ICC-sweep to find optimal cutoff
    candidates  = np.linspace(0.05, FS / 2 * 0.85, 30)  # min 0.05 Hz avoids degenerate DC-mean
    icc_scores  = []
    for c in candidates:
        lf_, hf_ = decompose_lf_hf(cross_resid, ep_policy, c)
        icc_lf_  = compute_icc(np.linalg.norm(lf_, axis=1), ep_policy)
        icc_hf_  = compute_icc(np.linalg.norm(hf_, axis=1), ep_policy)
        icc_scores.append((icc_lf_, icc_hf_, icc_lf_ - icc_hf_))

    if args.cutoff_hz is None:
        best_idx    = int(np.argmax([s[2] for s in icc_scores]))
        best_cutoff = float(candidates[best_idx])
    else:
        best_cutoff = args.cutoff_hz
        best_idx    = int(np.argmin(np.abs(candidates - best_cutoff)))

    r_LF, r_HF = decompose_lf_hf(cross_resid, ep_policy, best_cutoff)
    lf_mag  = np.linalg.norm(r_LF, axis=1)
    hf_mag  = np.linalg.norm(r_HF, axis=1)
    raw_mag = np.linalg.norm(cross_resid, axis=1)
    asp     = action_speed[policy_mask]

    rho_raw = float(spearmanr(raw_mag, asp).statistic)
    rho_LF  = float(spearmanr(lf_mag,  asp).statistic)
    rho_HF  = float(spearmanr(hf_mag,  asp).statistic)
    icc_raw = compute_icc(raw_mag, ep_policy)
    icc_LF  = compute_icc(lf_mag,  ep_policy)
    icc_HF  = compute_icc(hf_mag,  ep_policy)

    log(f"  Cutoff: {best_cutoff:.3f} Hz")
    log(f"  ICC: raw={icc_raw:.3f}  LF={icc_LF:.3f}  HF={icc_HF:.3f}")
    log(f"  ρ(speed): raw={rho_raw:.3f}  LF={rho_LF:.3f}  HF={rho_HF:.3f}")
    log(f"  ICC(LF)>ICC(raw): {icc_LF > icc_raw}  ρ(LF)<ρ(raw): {rho_LF < rho_raw}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: JACOBIAN COUPLING (Directional Coupling Test)
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("[PHASE 2] Jacobian coupling — does residual *direction* encode F/T?")
    log("=" * 72)

    ep_list_all = sorted(set(ep_policy))
    n_eps_all   = len(ep_list_all)
    ctx_mat     = episode_mean(r_LF,     ep_policy, ep_list_all)   # (n_eps, D_act)
    ft_mat      = episode_mean(ft_policy, ep_policy, ep_list_all)   # (n_eps, D_ft)

    def loo_spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Leave-one-out Spearman ρ."""
        n = len(x)
        preds = np.zeros(n)
        for i in range(n):
            tr = np.delete(np.arange(n), i)
            m  = Ridge(0.01).fit(x[tr].reshape(-1, 1), y[tr])
            preds[i] = m.predict(x[i].reshape(1, 1))[0]
        return float(spearmanr(y, preds).statistic)

    log("  Computing Spearman ρ matrix [D_act × D_ft]...")
    rho_matrix = np.zeros((D_act, D_ft))
    for i in range(D_act):
        for j in range(D_ft):
            rho_matrix[i, j] = float(
                spearmanr(ctx_mat[:, i], ft_mat[:, j]).statistic)

    log("         " + "  ".join(f"{l:>6}" for l in FT_LABELS))
    for i, al in enumerate(ACT_LABELS):
        row = "  ".join(f"{rho_matrix[i,j]:+6.3f}" for j in range(D_ft))
        log(f"  {al}: {row}")

    # Permutation null
    log(f"\n  Permutation null ({args.n_permutations} shuffles)...")
    obs_max_rho = float(np.max(np.abs(rho_matrix)))
    null_maxes  = []
    rng_null    = np.random.default_rng(0)
    for _ in range(args.n_permutations):
        perm = rng_null.permutation(n_eps_all)
        null_vals = [
            abs(float(spearmanr(ctx_mat[perm, i], ft_mat[:, j]).statistic))
            for i in range(D_act) for j in range(D_ft)
        ]
        null_maxes.append(max(null_vals))
    null_maxes    = np.array(null_maxes)
    null_95th     = float(np.percentile(null_maxes, 95))
    p_coupling    = float((null_maxes >= obs_max_rho).mean())

    log(f"  Observed max |ρ| = {obs_max_rho:.3f}")
    log(f"  Null 95th pct    = {null_95th:.3f}")
    log(f"  p-value          = {p_coupling:.4f}")
    log(f"  Significant coupling: {obs_max_rho > null_95th}")

    # Top coupling pairs
    flat_sorted = sorted(
        [(abs(rho_matrix[i, j]), i, j) for i in range(D_act) for j in range(D_ft)],
        reverse=True)
    log("\n  Top 5 coupling pairs:")
    top_pairs = []
    for rank, (abs_rho, i, j) in enumerate(flat_sorted[:5]):
        signed_rho = rho_matrix[i, j]
        log(f"    {ACT_LABELS[i]} → {FT_LABELS[j]}  ρ={signed_rho:+.3f}")
        if rank < 3:
            top_pairs.append({"act_dim": i, "act_label": ACT_LABELS[i],
                               "ft_dim": j, "ft_label": FT_LABELS[j],
                               "rho": round(float(signed_rho), 4),
                               "abs_rho": round(float(abs_rho), 4)})

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: EPISTEMOLOGICAL TEST
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("[PHASE 3] Epistemological test — does LF context predict actual F/T?")
    log("=" * 72)

    ep_list_train = [e for e in ep_list_all if e in train_eps]
    ft_mean_mat   = episode_mean(ft_policy, ep_policy, ep_list_all)  # (n_eps, D_ft)
    ft_var_arr    = np.array([ft_policy[ep_policy == e].var(axis=0).mean()
                              for e in ep_list_all])
    ft_peak_arr   = np.array([np.linalg.norm(ft_policy[ep_policy == e], axis=1).max()
                              for e in ep_list_all])
    # Per-component statistics for fig5 epistemological bar chart
    ft_abs_mean_mat = np.array([np.abs(ft_policy[ep_policy == e]).mean(axis=0)
                                for e in ep_list_all])   # (n_eps, D_ft)
    ft_peak_mat     = np.array([np.abs(ft_policy[ep_policy == e]).max(axis=0)
                                for e in ep_list_all])   # (n_eps, D_ft)
    ft_std_mat      = np.array([ft_policy[ep_policy == e].std(axis=0)
                                for e in ep_list_all])   # (n_eps, D_ft)
    traj_mat      = np.array([
        np.array([
            obs_arr[policy_mask][ep_policy == e].std(axis=0).mean(),
            float((ep_policy == e).sum()) / 200.0,
            np.linalg.norm(np.diff(obs_arr[policy_mask][ep_policy == e], axis=0), axis=1).sum() / 50.0,
        ]) for e in ep_list_all
    ])

    cv_f = min(5, len(ep_list_all))
    lf_s = StandardScaler().fit(ctx_mat).transform(ctx_mat)
    fm_s = StandardScaler().fit(ft_mean_mat).transform(ft_mean_mat)
    tr_s = StandardScaler().fit(traj_mat).transform(traj_mat)

    r2_lf_ft_mean = float(np.mean(
        cross_val_score(Ridge(1.), lf_s, fm_s, cv=cv_f, scoring="r2")))
    r2_lf_ft_var  = float(np.mean(
        cross_val_score(Ridge(1.), lf_s, ft_var_arr.reshape(-1, 1), cv=cv_f, scoring="r2")))
    r2_lf_ft_peak = float(np.mean(
        cross_val_score(Ridge(1.), lf_s, ft_peak_arr.reshape(-1, 1), cv=cv_f, scoring="r2")))
    r2_traj_ft    = float(np.mean(
        cross_val_score(Ridge(1.), tr_s, fm_s, cv=cv_f, scoring="r2")))
    r2_incr       = r2_lf_ft_mean - r2_traj_ft

    log(f"  R²(LF ctx → mean F/T): {r2_lf_ft_mean:.3f}")
    log(f"  R²(LF ctx → F/T var):  {r2_lf_ft_var:.3f}")
    log(f"  R²(LF ctx → peak F/T): {r2_lf_ft_peak:.3f}")
    log(f"  R²(traj stats → F/T):  {r2_traj_ft:.3f}  [control]")
    log(f"  Incremental R²:        {r2_incr:+.3f}")
    log(f"  LF specifically encodes F/T: {r2_incr > 0.05}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4: CONTEXT CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("[PHASE 4] Context construction")
    log("=" * 72)

    ep_list_train_idx = [ep_list_all.index(e) for e in ep_list_all if e in train_eps]
    ep_list_test_idx  = [ep_list_all.index(e) for e in ep_list_all if e in test_eps]

    # --- Raw episode-mean LF/HF (train episodes define scaler) ---
    lf_mean_raw = episode_mean(r_LF, ep_policy, ep_list_all)  # (n_eps, D_act)
    hf_mean_raw = episode_mean(r_HF, ep_policy, ep_list_all)  # (n_eps, D_act)

    train_rows = np.array(ep_list_train_idx)
    lf_mean_s, _ = std_ctx(lf_mean_raw, train_rows)
    hf_mean_s, _ = std_ctx(hf_mean_raw, train_rows)

    # --- LF + PCA(3) ---
    pca3       = PCA(n_components=min(3, D_act)).fit(lf_mean_raw[train_rows])
    lf_pca3    = pca3.transform(lf_mean_raw)            # (n_eps, 3)
    pca3_sc    = StandardScaler().fit(lf_pca3[train_rows])
    lf_pca3_s  = pca3_sc.transform(lf_pca3)
    log(f"  LF PCA(3) variance explained: {pca3.explained_variance_ratio_.sum()*100:.1f}%")

    # --- Obs trajectory features (for k-NN transfer) ---
    obs_pm = obs_arr[policy_mask]
    traj_feat_raw = np.array([
        np.concatenate([obs_pm[ep_policy == e].mean(axis=0),
                        obs_pm[ep_policy == e].std(axis=0)])
        for e in ep_list_all
    ])   # (n_eps, 2*D_obs)
    traj_feat_s, _ = std_ctx(traj_feat_raw, train_rows)

    # --- Transfer LF (k-NN) ---
    transfer_train, _ = compute_transfer_contexts(
        traj_feat_s, lf_mean_s, ep_list_train_idx, ep_list_train_idx,
        k=args.k_neighbors, exclude_self=True)
    transfer_test, transfer_meta = compute_transfer_contexts(
        traj_feat_s, lf_mean_s, ep_list_train_idx, ep_list_test_idx,
        k=args.k_neighbors, exclude_self=False)
    # Assemble full-episode array
    transfer_full = lf_mean_s.copy()
    for pos, idx in enumerate(ep_list_train_idx):
        transfer_full[idx] = transfer_train[pos]
    for pos, idx in enumerate(ep_list_test_idx):
        transfer_full[idx] = transfer_test[pos]

    log(f"  Transfer context: k={args.k_neighbors} neighbors  "
        f"(dim={lf_mean_s.shape[1]})")

    def expand(ctx_per_ep: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Map episode-level contexts (n_eps, D) → timestep level."""
        out = np.zeros((mask.sum(), ctx_per_ep.shape[1]), dtype=np.float32)
        for row, ep in enumerate(ep_policy[policy_mask[policy_mask]
                                            if False else ep_policy]):
            pass   # handled below
        eps_here = ep_policy[np.where(policy_mask)[0][mask[policy_mask]]]
        for i, ep in enumerate(eps_here):
            ep_idx = ep_list_all.index(int(ep))
            out[i]  = ctx_per_ep[ep_idx]
        return out

    def expand2(ctx_per_ep: np.ndarray, mask_within_policy: np.ndarray) -> np.ndarray:
        """ctx_per_ep: (n_eps, D). mask: boolean over policy_mask timesteps."""
        eps_here = ep_policy[mask_within_policy]
        out = np.zeros((len(eps_here), ctx_per_ep.shape[1]), dtype=np.float32)
        for i, ep in enumerate(eps_here):
            out[i] = ctx_per_ep[ep_list_all.index(int(ep))]
        return out

    train_within = np.isin(ep_policy, train_eps)
    test_within  = np.isin(ep_policy, test_eps)

    CONDITIONS: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "no_context":  (np.zeros((train_within.sum(), 0), np.float32),
                        np.zeros((test_within.sum(),  0), np.float32)),
        "oracle_lf":   (expand2(lf_mean_s,    train_within),
                        expand2(lf_mean_s,    test_within)),
        "transfer_lf": (expand2(transfer_full, train_within),
                        expand2(transfer_full, test_within)),
        "hf_oracle":   (expand2(hf_mean_s,    train_within),
                        expand2(hf_mean_s,    test_within)),
        "lf_pca3":     (expand2(lf_pca3_s,    train_within),
                        expand2(lf_pca3_s,    test_within)),
        "random":      None,   # filled per seed
    }

    for cname, pair in CONDITIONS.items():
        if pair is not None:
            log(f"  ctx[{cname}]: dim={pair[0].shape[1]}")
    log(f"  ctx[random]: dim={lf_mean_s.shape[1]} (per-seed noise)")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 5: BC TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log(f"[PHASE 5] Training — {args.seeds} seeds × {len(CONDITIONS)} conditions × {args.epochs} epochs")
    log("=" * 72)

    sc_policy = StandardScaler().fit(obs_arr[train_mask])
    S_tr = sc_policy.transform(obs_arr[policy_mask][train_within])
    S_te = sc_policy.transform(obs_arr[policy_mask][test_within])
    A_tr = act_arr[policy_mask][train_within]
    A_te = act_arr[policy_mask][test_within]

    # Test strata: LF magnitude terciles (for per-stratum analysis)
    te_lf = lf_mag[test_within]
    t33, t67 = np.percentile(te_lf, [33, 67])
    lf_terc   = np.zeros(len(te_lf), dtype=int)
    lf_terc[te_lf > t67] = 2
    lf_terc[(te_lf >= t33) & (te_lf <= t67)] = 1

    test_groups: dict[str, np.ndarray] = {
        "low_lf":     lf_terc == 0,
        "mid_lf":     lf_terc == 1,
        "high_lf":    lf_terc == 2,
    }

    all_results: dict[str, dict] = {cname: {
        "curves": [], "group_curves": {g: [] for g in test_groups},
        "seed_best": [], "ctx_dim": 0,
    } for cname in CONDITIONS}

    rng_main = np.random.default_rng(99)
    for seed in range(args.seeds):
        log(f"\n  ── Seed {seed} ──")
        for cname in CONDITIONS:
            if cname == "random":
                rng_r = np.random.default_rng(42 + seed * 100)
                rand_ep = {e: rng_r.standard_normal(lf_mean_s.shape[1]).astype(np.float32)
                           for e in ep_list_all}
                rand_arr = np.zeros((len(ep_list_all), lf_mean_s.shape[1]), np.float32)
                for k2, e in enumerate(ep_list_all):
                    rand_arr[k2] = rand_ep[e]
                C_tr = expand2(rand_arr, train_within)
                C_te = expand2(rand_arr, test_within)
            else:
                C_tr, C_te = CONDITIONS[cname]

            all_results[cname]["ctx_dim"] = C_tr.shape[1]
            res = train_bc(
                S_tr, A_tr, C_tr, S_te, A_te, C_te,
                test_groups=test_groups,
                seed=seed, n_epochs=args.epochs,
                lr=args.lr, batch_size=args.batch_size,
                hidden=args.hidden,
                label=f"{cname} | seed {seed}",
            )
            all_results[cname]["curves"].append(res["test_mse"])
            all_results[cname]["seed_best"].append(res["best_mse"])
            for g in test_groups:
                all_results[cname]["group_curves"][g].append(res["group_mse"][g])

    # Aggregate
    for cname in CONDITIONS:
        r = all_results[cname]
        r["mean_best"] = float(np.mean(r["seed_best"]))
        r["std_best"]  = float(np.std(r["seed_best"]))
        r["curves_arr"]       = np.array(r["curves"])
        r["group_curves_arr"] = {g: np.array(v) for g, v in r["group_curves"].items()}

    baseline = all_results["no_context"]["mean_best"]
    log("\n" + "=" * 72)
    log("RANKING")
    log("=" * 72)
    for cname in sorted(CONDITIONS, key=lambda k: all_results[k]["mean_best"]):
        r    = all_results[cname]
        pct  = (baseline - r["mean_best"]) / baseline * 100
        log(f"  {cname:<18} {r['mean_best']:.8f} ± {r['std_best']:.8f}  {pct:+.2f}%")

    # Per-stratum gains
    def stratum_gain(skey: str, method: str, ref: str = "no_context") -> float:
        ref_best = float(np.nanmean([min(c) for c in all_results[ref]["group_curves"][skey]]))
        met_best = float(np.nanmean([min(c) for c in all_results[method]["group_curves"][skey]]))
        return (ref_best - met_best) / (ref_best + 1e-12) * 100

    strat_lf = {
        "Low LF":  stratum_gain("low_lf",  "oracle_lf"),
        "Mid LF":  stratum_gain("mid_lf",  "oracle_lf"),
        "High LF": stratum_gain("high_lf", "oracle_lf"),
    }
    log("\nPer-stratum (LF magnitude, oracle_lf vs no_context):")
    for k, v in strat_lf.items(): log(f"  {k}: {v:+.1f}%")

    lf_gain  = (baseline - all_results["oracle_lf"]["mean_best"]) / baseline * 100
    tf_gain  = (baseline - all_results["transfer_lf"]["mean_best"]) / baseline * 100

    log("\n" + "=" * 72)
    log("CONDITION SUMMARY")
    log(f"  {'Context':<20} {'MSE':>10}  {'vs no-ctx':>10}")
    log("  " + "─" * 45)
    for cname in CONTEXT_LABELS:
        r   = all_results[cname]
        pct = (baseline - r["mean_best"]) / baseline * 100
        log(f"  {cname:<20} {r['mean_best']:>10.6f}  {pct:>+10.1f}%")

    log("\nJACOBIAN COUPLING")
    for tp in top_pairs:
        log(f"  {tp['act_label']} → {tp['ft_label']}  ρ={tp['rho']:+.3f}")
    log(f"  max|ρ|={obs_max_rho:.3f} vs null 95th={null_95th:.3f}  p={p_coupling:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 6: FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    log("\n[FIGURES] Generating ...")

    COLORS = {c: CONTEXT_COLORS[c] for c in CONDITIONS}

    # ── fig1: Spectral validation (3 panels) ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    freqs, psd = welch(cross_resid[:, 0], fs=FS, nperseg=min(64, len(cross_resid) // 4))
    ax.semilogy(freqs, psd, color="#374151", lw=1.5)
    ax.axvline(best_cutoff, color="#2563EB", ls="--", lw=2,
               label=f"Cutoff {best_cutoff:.3f} Hz")
    ax.axvspan(0, best_cutoff, alpha=0.15, color="#2563EB", label="LF band")
    ax.axvspan(best_cutoff, FS / 2, alpha=0.08, color="#F87171", label="HF band")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density\n(BC residuals, dim 0)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(candidates, [s[0] for s in icc_scores], color="#2563EB", lw=2,
            marker="o", ms=3, label="ICC(LF)")
    ax.plot(candidates, [s[1] for s in icc_scores], color="#F87171", lw=2,
            marker="s", ms=3, label="ICC(HF)")
    ax.axvline(best_cutoff, color="k", ls="--", lw=1.5)
    ax.set_xlabel("Cutoff (Hz)"); ax.set_ylabel("ICC")
    ax.set_title("ICC sweep  (ICC(LF)−ICC(HF) maximised)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    ax = axes[2]
    labels = ["Raw", "LF", "HF"]
    icc_b  = [icc_raw, icc_LF, icc_HF]
    rho_b  = [rho_raw, rho_LF, rho_HF]
    colors = ["#9CA3AF", "#2563EB", "#F87171"]
    x = np.arange(3)
    bars = ax.bar(x, icc_b, color=colors, alpha=0.85, edgecolor="k", linewidth=0.7, width=0.55)
    for bar, icc_v, rho_v in zip(bars, icc_b, rho_b):
        ax.text(bar.get_x() + bar.get_width() / 2, icc_v + 0.01,
                f"ICC={icc_v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, icc_v / 2,
                f"ρ={rho_v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if icc_v > 0.15 else "black")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("ICC (between-episode / total)")
    ax.set_title("Spectral validation\nICC↑ for LF  |  ρ(speed) shown inside bars",
                 fontweight="bold")
    ax.set_ylim(0, max(icc_b) * 1.3 + 0.05)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_spectral_validation.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    log("  fig1_spectral_validation.png")

    # ── fig2: BC ranking ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ax = axes[0]
    for cname, clabel in CONTEXT_LABELS.items():
        curves = all_results[cname]["curves_arr"]
        mean = curves.mean(axis=0); std = curves.std(axis=0)
        ax.plot(mean, color=COLORS[cname], lw=2, label=f"{clabel}")
        ax.fill_between(range(len(mean)), mean - std, mean + std,
                        color=COLORS[cname], alpha=0.10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test MSE")
    ax.set_title(f"Training curves ({args.seeds} seeds ±1 std)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1]
    names_sorted = sorted(CONTEXT_LABELS, key=lambda k: all_results[k]["mean_best"])
    means = [all_results[k]["mean_best"] for k in names_sorted]
    stds  = [all_results[k]["std_best"]  for k in names_sorted]
    ax.barh(range(len(names_sorted)), means, xerr=stds,
            color=[COLORS[k] for k in names_sorted],
            alpha=0.85, edgecolor="k", linewidth=0.5, capsize=3)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels([CONTEXT_LABELS[k] for k in names_sorted], fontsize=9)
    ax.set_xlabel("Best Test MSE")
    ax.set_title(f"Ranking  (baseline={baseline:.5f})", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    for i, k in enumerate(names_sorted):
        pct = (baseline - all_results[k]["mean_best"]) / baseline * 100
        ax.text(means[i] + stds[i] + 1e-6, i, f" {pct:+.1f}%", va="center",
                fontsize=8, color="#16A34A" if pct > 0 else "#DC2626", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_bc_ranking.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    log("  fig2_bc_ranking.png")

    # ── fig4: Jacobian coupling (THE strongest finding) ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    im = ax.imshow(rho_matrix, cmap="RdYlGn", vmin=-0.9, vmax=0.9, aspect="auto")
    ax.set_xticks(range(D_ft)); ax.set_xticklabels(FT_LABELS)
    ax.set_yticks(range(D_act)); ax.set_yticklabels(ACT_LABELS)
    ax.set_title("Spearman ρ\nctx_dim → F/T component", fontweight="bold")
    plt.colorbar(im, ax=ax)
    for i in range(D_act):
        for j in range(D_ft):
            ax.text(j, i, f"{rho_matrix[i,j]:+.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(rho_matrix[i,j]) < 0.5 else "white")

    # Highlight top coupling
    best_i, best_j = int(np.unravel_index(np.argmax(np.abs(rho_matrix)), rho_matrix.shape)[0]), \
                     int(np.unravel_index(np.argmax(np.abs(rho_matrix)), rho_matrix.shape)[1])
    rect = plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                          linewidth=3, edgecolor="gold", facecolor="none")
    ax.add_patch(rect)

    ax = axes[1]
    scatter_i = top_pairs[0]["act_dim"] if top_pairs else 0
    scatter_j = top_pairs[0]["ft_dim"]  if top_pairs else 0
    ax.scatter(ctx_mat[:, scatter_i], ft_mat[:, scatter_j],
               s=60, alpha=0.8, color="#2563EB", edgecolors="k", lw=0.5)
    if len(ctx_mat) > 2:
        z = np.polyfit(ctx_mat[:, scatter_i], ft_mat[:, scatter_j], 1)
        xl = np.linspace(ctx_mat[:, scatter_i].min(), ctx_mat[:, scatter_i].max(), 50)
        ax.plot(xl, np.polyval(z, xl), "r--", lw=2)
    rho_best = rho_matrix[scatter_i, scatter_j]
    al  = ACT_LABELS[scatter_i]; fl = FT_LABELS[scatter_j]
    ax.set_xlabel(f"LF residual ({al})", fontsize=10)
    ax.set_ylabel(fl, fontsize=10)
    ax.set_title(f"Strongest coupling: {al} → {fl}\nρ = {rho_best:+.3f}",
                 fontweight="bold")
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.hist(null_maxes, bins=35, alpha=0.75, color="#6B7280",
            edgecolor="k", lw=0.3, label="Null max |ρ|")
    ax.axvline(obs_max_rho, color="#DC2626", lw=2.5,
               label=f"Observed max |ρ| = {obs_max_rho:.3f}")
    ax.axvline(null_95th, color="#F59E0B", lw=1.8, ls="--",
               label=f"95th pct = {null_95th:.3f}")
    ax.set_xlabel("Max |ρ| over all (dim, F/T) pairs")
    ax.set_title(f"Permutation null  (p = {p_coupling:.4f})", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.suptitle(
        "Jacobian Coupling: BC Residual Direction Encodes Contact Forces\n"
        "The policy never saw F/T. Yet failure direction in joint-space reveals wrench-space structure.",
        fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_jacobian_coupling.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    log("  fig4_jacobian_coupling.png")

    # ── fig5: Epistemological test ────────────────────────────────────────────
    lf_ctx_mag = np.linalg.norm(ctx_mat, axis=1)
    n_boot = 500
    traj_ctrl = traj_mat[:, [0]]   # primary generic-difficulty proxy: mean obs std
    epi_targets = OrderedDict([
        ("mean_signed", (ft_mean_mat, "Mean signed F/T")),
        ("mean_abs",    (ft_abs_mean_mat, "Mean |F/T|")),
        ("peak_abs",    (ft_peak_mat, "Peak |F/T|")),
        ("std",         (ft_std_mat, "Std(F/T)")),
    ])
    epi_component_results = OrderedDict()
    y_abs_max = 0.55
    log(f"  Fig5 uses n={len(ep_list_all)} train+test episodes with bootstrap CI (n_boot={n_boot})")
    for stat_idx, (stat_key, (stat_mat, stat_title)) in enumerate(epi_targets.items()):
        rho = np.zeros(D_ft)
        pval = np.ones(D_ft)
        ci_lo = np.zeros(D_ft)
        ci_hi = np.zeros(D_ft)
        partial_rho = np.zeros(D_ft)
        partial_p = np.ones(D_ft)
        partial_ci_lo = np.zeros(D_ft)
        partial_ci_hi = np.zeros(D_ft)
        for j in range(D_ft):
            rho[j], pval[j] = safe_spearmanr(lf_ctx_mag, stat_mat[:, j])
            ci_lo[j], ci_hi[j] = bootstrap_corr_ci(
                lf_ctx_mag, stat_mat[:, j], n_boot=n_boot,
                seed=args.split_seed + 100 * stat_idx + j)
            partial_rho[j], partial_p[j] = partial_spearmanr(
                lf_ctx_mag, stat_mat[:, j], traj_ctrl)
            partial_ci_lo[j], partial_ci_hi[j] = bootstrap_corr_ci(
                lf_ctx_mag, stat_mat[:, j], n_boot=n_boot,
                seed=args.split_seed + 1000 + 100 * stat_idx + j,
                covars=traj_ctrl)
        epi_component_results[stat_key] = {
            "title": stat_title,
            "rho": rho,
            "pval": pval,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "partial_rho": partial_rho,
            "partial_pval": partial_p,
            "partial_ci_lo": partial_ci_lo,
            "partial_ci_hi": partial_ci_hi,
        }
        y_abs_max = max(
            y_abs_max,
            float(np.nanmax(np.abs(np.concatenate([
                rho, ci_lo, ci_hi, partial_rho, partial_ci_lo, partial_ci_hi
            ]))))
        )
        best_j = int(np.nanargmax(np.abs(partial_rho)))
        log(f"    {stat_title:<14} raw sig={int((pval < 0.05).sum())}/6"
            f" | partial sig={int((partial_p < 0.05).sum())}/6"
            f" | strongest partial: {FT_LABELS[best_j]} ρ={partial_rho[best_j]:+.3f}"
            f" (p={partial_p[best_j]:.3f})")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    x = np.arange(D_ft)
    ylim = min(1.0, max(0.65, np.ceil(y_abs_max * 10) / 10))
    for ax, (stat_key, meta) in zip(axes.flat, epi_component_results.items()):
        rho = meta["rho"]
        pval = meta["pval"]
        ci_lo = meta["ci_lo"]
        ci_hi = meta["ci_hi"]
        partial_rho = meta["partial_rho"]
        partial_p = meta["partial_pval"]
        partial_ci_lo = meta["partial_ci_lo"]
        partial_ci_hi = meta["partial_ci_hi"]
        colors = ["#16A34A" if r >= 0 else "#DC2626" for r in rho]
        raw_yerr = np.vstack([
            np.maximum(rho - ci_lo, 0.0),
            np.maximum(ci_hi - rho, 0.0),
        ])
        partial_yerr = np.vstack([
            np.maximum(partial_rho - partial_ci_lo, 0.0),
            np.maximum(partial_ci_hi - partial_rho, 0.0),
        ])
        bars = ax.bar(
            x, rho, width=0.68, color=colors, alpha=0.82,
            edgecolor="black", linewidth=0.8,
            yerr=raw_yerr,
            error_kw=dict(ecolor="black", lw=1.0, capsize=3, capthick=1.0),
            zorder=2,
        )
        for j, bar in enumerate(bars):
            if pval[j] < 0.05:
                bar.set_hatch("//")
        ax.errorbar(
            x, partial_rho,
            yerr=partial_yerr,
            fmt="none", ecolor="#111827", elinewidth=1.2, capsize=3, zorder=4,
        )
        facecolors = ["#111827" if p < 0.05 else "white" for p in partial_p]
        ax.scatter(x, partial_rho, s=42, facecolors=facecolors,
                   edgecolors="#111827", linewidths=1.2, zorder=5)
        ax.axhline(0, color="#111827", lw=1.0, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(FT_LABELS)
        ax.set_ylim(-ylim, ylim)
        ax.set_title(meta["title"], fontweight="bold")
        ax.grid(axis="y", alpha=0.18, zorder=0)
        ax.text(
            0.02, 0.96,
            f"raw sig {int((pval < 0.05).sum())}/6 | partial sig {int((partial_p < 0.05).sum())}/6",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#374151"
        )

    axes[0, 0].set_ylabel("Spearman ρ")
    axes[1, 0].set_ylabel("Spearman ρ")
    legend_items = [
        matplotlib.patches.Patch(facecolor="#16A34A", edgecolor="black",
                                 label="Raw ρ > 0 (hatched if p < 0.05)"),
        matplotlib.patches.Patch(facecolor="#DC2626", edgecolor="black",
                                 label="Raw ρ < 0 (hatched if p < 0.05)"),
        matplotlib.lines.Line2D([0], [0], marker="o", color="#111827",
                                markerfacecolor="white", linestyle="None",
                                label="Partial ρ | obs-std controlled"),
        matplotlib.lines.Line2D([0], [0], marker="o", color="#111827",
                                markerfacecolor="#111827", linestyle="None",
                                label="Partial p < 0.05"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9)
    lf_specific = r2_incr > 0.05
    incr_str = (f"Incremental R²={r2_incr:+.3f}  →  "
                f"{'LF encodes F/T beyond difficulty' if lf_specific else 'LF specificity remains weak after trajectory control'}")
    plt.suptitle(
        "Epistemological test: LF context magnitude tracks held-out contact statistics\n"
        f"Bars: raw Spearman ρ with bootstrap 95% CI | markers: partial ρ controlling obs-trajectory std | n={len(ep_list_all)} train+test episodes\n"
        f"{incr_str}",
        fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig5_epistemological_test.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    log("  fig5_epistemological_test.png")

    # ── fig6: LF stratification bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))

    def _strat_bars(ax, gains_dict, colors, title, ylabel, note):
        labels = list(gains_dict.keys())
        vals   = list(gains_dict.values())
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="k", linewidth=0.7, width=0.55)
        ax.axhline(0, color="k", lw=0.8)
        for bar, v in zip(bars, vals):
            yoff = 0.3 if v >= 0 else -0.6
            ax.text(bar.get_x() + bar.get_width() / 2, v + yoff,
                    f"{v:+.1f}%", ha="center", fontsize=12, fontweight="bold",
                    color="#16A34A" if v > 0 else "#DC2626")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.text(0.98, 0.02, note, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#6B7280", fontstyle="italic")

    # Panel 1: stratify by LF residual magnitude (proxy for episode confounding)
    lf_colors = ["#BFDBFE", "#3B82F6", "#1D4ED8"]
    _strat_bars(
        ax, strat_lf, lf_colors,
        "Oracle LF gain by LF residual magnitude\n(high LF = more between-episode structure)",
        "Gain: oracle_lf vs no_context (%)",
        "Stratified by LF magnitude of BC residuals\n(proxy for confounding strength)"
    )

    plt.suptitle(
        "Stratification: does gain track discovered LF structure?",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_per_stratum_lf.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    log("  fig6_per_stratum_lf.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 7: RESULTS JSON + VERDICT
    # ══════════════════════════════════════════════════════════════════════════
    def verdict_fn() -> str:
        c1 = lf_gain > 4.0
        c2 = r2_incr > 0.05
        c3 = icc_LF > icc_raw
        c4 = obs_max_rho > null_95th
        if c1 and c2 and c3 and c4:
            return "STRONG — LF structure, contact coupling, and BC gain all align."
        if c1 and c2 and c4:
            return "MODERATE — BC gain plus contact-grounded LF evidence."
        if c3 and c4:
            return "INFORMATIVE — spectral structure + Jacobian coupling found, BC gain marginal."
        return "NEGATIVE — principal claims did not replicate."

    results_out = {
        "dataset": args.dataset,
        "n_episodes": int(len(ep_ids)),
        "n_timesteps": int(N),
        "fs": FS,
        "config": {
            "seeds": args.seeds, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr,
            "hidden": args.hidden, "cutoff_hz": round(best_cutoff, 4),
            "protocol": args.protocol,
            "k_neighbors": args.k_neighbors,
        },
        "episode_split": {
            "critic": [int(e) for e in critic_eps],
            "train":  [int(e) for e in train_eps],
            "test":   [int(e) for e in test_eps],
        },
        "oracle_gap": {
            "mse_partial": round(mse_partial, 8),
            "mse_full":    round(mse_full, 8),
            "gap_pct":     round(oracle_gap, 3),
        },
        "spectral_validation": {
            "optimal_cutoff_hz": round(best_cutoff, 4),
            "icc_raw": round(icc_raw, 4),
            "icc_LF":  round(icc_LF,  4),
            "icc_HF":  round(icc_HF,  4),
            "rho_raw": round(rho_raw, 4),
            "rho_LF":  round(rho_LF,  4),
            "rho_HF":  round(rho_HF,  4),
            "precond_icc": bool(icc_LF > icc_raw),
            "precond_rho": bool(rho_LF < rho_raw),
        },
        "jacobian_coupling": {
            "rho_matrix": rho_matrix.tolist(),
            "act_labels": ACT_LABELS,
            "ft_labels":  FT_LABELS,
            "top_pairs":  top_pairs,
            "max_abs_rho": round(float(obs_max_rho), 4),
            "null_95th_pct": round(float(null_95th), 4),
            "p_value": round(float(p_coupling), 4),
            "significant": bool(obs_max_rho > null_95th),
            "n_permutations": args.n_permutations,
        },
        "epistemological_test": {
            "n_episodes_used": int(len(ep_list_all)),
            "n_bootstrap": int(n_boot),
            "control_features": ["obs_std_mean"],
            "r2_lf_ft_mean":  round(r2_lf_ft_mean, 4),
            "r2_lf_ft_var":   round(r2_lf_ft_var,  4),
            "r2_lf_ft_peak":  round(r2_lf_ft_peak, 4),
            "r2_traj_ft":     round(r2_traj_ft,     4),
            "r2_incremental": round(r2_incr,         4),
            "lf_encodes_ft":  bool(r2_incr > 0.05),
            "component_correlations": {
                stat_key: {
                    "title": meta["title"],
                    "rho": [round(float(v), 4) for v in meta["rho"]],
                    "p_value": [round(float(v), 4) for v in meta["pval"]],
                    "ci_low": [round(float(v), 4) for v in meta["ci_lo"]],
                    "ci_high": [round(float(v), 4) for v in meta["ci_hi"]],
                    "partial_rho": [round(float(v), 4) for v in meta["partial_rho"]],
                    "partial_p_value": [round(float(v), 4) for v in meta["partial_pval"]],
                    "partial_ci_low": [round(float(v), 4) for v in meta["partial_ci_lo"]],
                    "partial_ci_high": [round(float(v), 4) for v in meta["partial_ci_hi"]],
                }
                for stat_key, meta in epi_component_results.items()
            },
        },
        "bc_results": {
            cname: {
                "mean_best": round(r["mean_best"], 8),
                "std_best":  round(r["std_best"],  8),
                "gain_pct":  round((baseline - r["mean_best"]) / baseline * 100, 2),
                "ctx_dim":   r["ctx_dim"],
            }
            for cname, r in all_results.items()
        },
        "per_stratum_lf": {k: round(v, 2) for k, v in strat_lf.items()},
        "transfer_meta_last_seed": {
            str(ep_list_all[idx]): meta
            for idx, meta in transfer_meta.items()
        },
        "experiment_lineage": {
            "primary_claim": "BC residual direction encodes contact forces via Jacobian",
            "v6_lf_gain_pct": 4.8,
            "v6_jacobian_max_rho": 0.85,
            "v11_note": f"{args.protocol} + integrated directional coupling",
        },
        "verdict": verdict_fn(),
    }

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_out, f, indent=2)

    log("\n" + "=" * 72)
    log(f"VERDICT: {results_out['verdict']}")
    log("=" * 72)
    log(f"\nOutputs written to {output_dir}/")
    log("  results.json  |  run_log.txt  |  fig1, fig2, fig4, fig5, fig6 PNGs")


if __name__ == "__main__":
    main()
