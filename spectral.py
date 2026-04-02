"""
Spectral Context Extraction
============================
Cross-fitted residuals → ICC-optimized band-pass decomposition →
episode-level LF context vectors.

Reuses the validated approach from v5/v6 of the causal completeness series.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr


def compute_icc(scores, episode_ids):
    """One-way random-effects ICC (intra-class correlation)."""
    eps     = np.unique(episode_ids)
    grand_m = scores.mean()
    n_g     = len(eps)
    n_tot   = len(scores)
    g_sizes = np.array([np.sum(episode_ids == e) for e in eps])
    g_means = np.array([scores[episode_ids == e].mean() for e in eps])
    ss_bet  = np.sum(g_sizes * (g_means - grand_m) ** 2)
    ms_bet  = ss_bet / max(n_g - 1, 1)
    ss_with = sum(((scores[episode_ids == e] - g_means[i]) ** 2).sum()
                  for i, e in enumerate(eps))
    ms_with = ss_with / max(n_tot - n_g, 1)
    k0      = (n_tot - np.sum(g_sizes ** 2) / n_tot) / max(n_g - 1, 1)
    icc     = (ms_bet - ms_with) / (ms_bet + (k0 - 1) * ms_with + 1e-12)
    return float(max(0, icc))


def _lowpass_per_episode(residuals, episodes, b, a, min_len=17):
    """Apply Butterworth low-pass filter per episode."""
    r_LF = np.zeros_like(residuals)
    r_HF = np.zeros_like(residuals)
    for ep in np.unique(episodes):
        mask = episodes == ep
        r_ep = residuals[mask]
        if len(r_ep) < min_len:
            r_LF[mask] = r_ep.mean(axis=0)
            r_HF[mask] = r_ep - r_ep.mean(axis=0)
        else:
            for d in range(residuals.shape[1]):
                lf = filtfilt(b, a, r_ep[:, d])
                r_LF[mask, d] = lf
                r_HF[mask, d] = r_ep[:, d] - lf
    return r_LF, r_HF


def extract_spectral_context(S, A, episodes, fs=50, cutoff_hz=None,
                              ridge_alpha=1.0, n_splits=5):
    """
    Full spectral context extraction pipeline.

    Parameters
    ----------
    S          : (N, D_obs) observations
    A          : (N, D_act) actions
    episodes   : (N,) episode indices
    fs         : sampling frequency in Hz
    cutoff_hz  : if None, auto-selected by ICC sweep
    ridge_alpha: regularisation for cross-fitting
    n_splits   : GroupKFold folds

    Returns
    -------
    ctx_dict  : dict {ep_id → (D_act,) standardised context vector}
    metrics   : dict with ICC, rho, cutoff diagnostics
    cross_resid : (N, D_act) cross-fitted residuals
    r_LF      : (N, D_act) low-frequency component
    r_HF      : (N, D_act) high-frequency component
    obs_scaler: fitted StandardScaler for observations
    """
    obs_scaler = StandardScaler().fit(S)
    S_sc       = obs_scaler.transform(S)
    ep_ids     = np.sort(np.unique(episodes))

    # ── Cross-fitted residuals (DML-style, no circularity) ────────────────
    n_folds   = min(n_splits, len(ep_ids) // 2)
    gkf       = GroupKFold(n_splits=n_folds)
    cross_resid = np.zeros_like(A)
    for tr_idx, te_idx in gkf.split(S_sc, groups=episodes):
        m = Ridge(alpha=ridge_alpha).fit(S_sc[tr_idx], A[tr_idx])
        cross_resid[te_idx] = A[te_idx] - m.predict(S_sc[te_idx])

    # ── Action speed (confound for spectral validation) ───────────────────
    action_speed = np.zeros(len(A))
    for ep in ep_ids:
        idx = np.where(episodes == ep)[0]
        if len(idx) > 1:
            action_speed[idx[1:]] = np.linalg.norm(
                np.diff(A[idx], axis=0), axis=1)

    # ── ICC-optimised cutoff selection ────────────────────────────────────
    if cutoff_hz is None:
        nyq        = fs / 2.0
        candidates = [c for c in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
                      if c < nyq]
        best_score, best_cutoff = -np.inf, candidates[0]
        for c in candidates:
            b, a = butter(4, c / nyq, btype='low')
            lf_, _ = _lowpass_per_episode(cross_resid, episodes, b, a)
            lf_m   = np.linalg.norm(lf_, axis=1)
            hf_m   = np.linalg.norm(cross_resid - lf_, axis=1)
            icc_lf = compute_icc(lf_m, episodes)
            icc_hf = compute_icc(hf_m, episodes)
            rho_lf = abs(float(spearmanr(lf_m, action_speed).statistic))
            score  = (icc_lf - icc_hf) * (1.0 - rho_lf)
            if score > best_score:
                best_score, best_cutoff = score, c
        cutoff_hz = best_cutoff

    # ── Decompose at optimal cutoff ───────────────────────────────────────
    nyq  = fs / 2.0
    b, a = butter(4, cutoff_hz / nyq, btype='low')
    r_LF, r_HF = _lowpass_per_episode(cross_resid, episodes, b, a)

    # ── Diagnostics ───────────────────────────────────────────────────────
    lf_m   = np.linalg.norm(r_LF,       axis=1)
    hf_m   = np.linalg.norm(r_HF,       axis=1)
    raw_m  = np.linalg.norm(cross_resid, axis=1)

    metrics = {
        'cutoff_hz' : float(cutoff_hz),
        'icc_LF'    : compute_icc(lf_m,  episodes),
        'icc_HF'    : compute_icc(hf_m,  episodes),
        'icc_raw'   : compute_icc(raw_m, episodes),
        'rho_LF'    : float(spearmanr(lf_m,  action_speed).statistic),
        'rho_HF'    : float(spearmanr(hf_m,  action_speed).statistic),
        'rho_raw'   : float(spearmanr(raw_m, action_speed).statistic),
    }

    # ── Episode-level LF context vectors (standardised mean) ─────────────
    lf_mat    = np.array([r_LF[episodes == e].mean(axis=0) for e in ep_ids])
    ctx_sc    = StandardScaler().fit(lf_mat)
    lf_mat_s  = ctx_sc.transform(lf_mat)
    ctx_dict  = {e: lf_mat_s[i] for i, e in enumerate(ep_ids)}

    return ctx_dict, metrics, cross_resid, r_LF, r_HF, obs_scaler
