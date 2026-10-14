"""Time-series helpers for records longer than 48 h (chained segments) + online trigger rule."""
import numpy as np
import pandas as pd
import torch

from ts_inference import VARIABLES, N_VARS, make_batch


def featurize_long(raw, stats, window):
    """Same features as ts_inference.featurize, for any number of hours. raw [T, V] -> [1, T, 3V]."""
    raw = np.asarray(raw, dtype=np.float64)[None]
    T = raw.shape[1]
    lo, hi = np.array(stats["clip_lo"]), np.array(stats["clip_hi"])
    mu, sd = np.array(stats["mean"]), np.array(stats["std"])
    mask = ~np.isnan(raw)
    z = (np.clip(raw, lo, hi) - mu) / sd
    t_idx = np.arange(T)[None, :, None]
    last = np.maximum.accumulate(np.where(mask, t_idx, -1), axis=1)
    ever = last >= 0
    z_ff = np.take_along_axis(np.nan_to_num(z, nan=0.0), np.clip(last, 0, None), axis=1)
    z_ff = np.where(ever, z_ff, 0.0)
    since = np.minimum(np.where(ever, t_idx - last, window), window) / window
    return np.concatenate([z_ff, mask.astype(float), since], axis=-1).astype(np.float32)


@torch.no_grad()
def p_ts_series(ts, raw, start, end):
    """Calibrated p_ts for windows ending at hours start..end (hour h uses data from hours < h)."""
    W = ts.cfg["window"]
    X = torch.tensor(featurize_long(raw, ts.stats, W), device=ts.device)
    ends = torch.arange(start, end + 1, device=ts.device)
    xb, pad = make_batch(X, torch.zeros_like(ends), ends, W)
    z = ts.model(xb, pad).float().cpu().numpy()
    p = ts.calibrate(z)
    return pd.DataFrame({"hour": ends.cpu().numpy(), "p_ts": p, "delta_ts": ts.delta(p)})


@torch.no_grad()
def explain_long(ts, raw, end_hour, top_k=4):
    """Occlusion on the window ending at end_hour (same method as VitalsRiskModel.explain)."""
    W = ts.cfg["window"]
    X = torch.tensor(featurize_long(raw, ts.stats, W), device=ts.device)
    xb, pad = make_batch(X, torch.tensor([0], device=ts.device), torch.tensor([end_hour], device=ts.device), W)
    V = N_VARS
    occl = xb.repeat(V + 1, 1, 1)
    for j in range(V):
        occl[j + 1, :, j] = 0.0; occl[j + 1, :, V + j] = 0.0; occl[j + 1, :, 2 * V + j] = 1.0
    occl = occl * (~pad).unsqueeze(-1).repeat(V + 1, 1, 1)
    z = ts.model(occl, pad.repeat(V + 1, 1)).float().cpu().numpy()
    contrib = z[0] - z[1:]
    win = np.asarray(raw)[max(0, end_hour - W):end_hour]
    rows = []
    for j, v in enumerate(VARIABLES):
        obs = win[:, j][~np.isnan(win[:, j])]
        trend = ""
        if len(obs) >= 2:
            trend = "rising" if obs[-1] > obs[0] else ("falling" if obs[-1] < obs[0] else "flat")
        rows.append({"variable": v, "contribution": float(contrib[j]), "n_obs": int(len(obs)),
                     "first": float(obs[0]) if len(obs) else np.nan, "last": float(obs[-1]) if len(obs) else np.nan,
                     "trend": trend})
    df = pd.DataFrame(rows)
    df = df[df.n_obs > 0]
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)


def latest_values(raw, hour, names):
    """Last observed value of each variable before `hour` (carried forward)."""
    out = {}
    for v in names:
        col = np.asarray(raw)[:hour, VARIABLES.index(v)]
        obs = np.where(~np.isnan(col))[0]
        out[v] = (float(col[obs[-1]]), int(hour - 1 - obs[-1])) if len(obs) else (np.nan, None)
    return out


class TriggerMonitor:
    """Online version of the Notebook-2 re-imaging rule (identical decisions to detect_triggers)."""

    def __init__(self, cfg, first_image_hour):
        self.c = cfg
        self.hist, self.run, self.last_image = [], 0, first_image_hour

    def step(self, hour, p):
        c = self.c
        self.hist.append(float(p))
        self.run = self.run + 1 if p >= c["theta"] else 0
        sustained = self.run >= c["k_consecutive"]
        m = c["rise_hours"]
        rise = len(self.hist) - 1 >= m and (p - self.hist[-1 - m]) >= c["rise_delta"] and p >= c.get("rise_floor", 0.0)
        if (sustained or rise) and hour - self.last_image >= c["cooldown_hours"]:
            self.last_image = hour
            return "sustained high + rapid rise" if (sustained and rise) else ("sustained high" if sustained else "rapid rise")
        return None
