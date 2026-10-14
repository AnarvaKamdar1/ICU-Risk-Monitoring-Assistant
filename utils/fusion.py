"""Composite risk index: baseline log-odds + capped, reliability-weighted adjustments."""
import numpy as np


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fuse(p_base, p_ts, p_img, alpha, beta, base_rate_ts, base_rate_img, cap=2.0):
    d_ts = float(logit(p_ts) - logit(base_rate_ts))
    d_img = float(logit(p_img) - logit(base_rate_img))
    raw_ts, raw_img = alpha * d_ts, beta * d_img
    c_ts, c_img = float(np.clip(raw_ts, -cap, cap)), float(np.clip(raw_img, -cap, cap))
    z = float(logit(p_base)) + c_ts + c_img
    disagree = abs(c_ts - c_img)
    opposite = (c_ts * c_img < 0) and min(abs(c_ts), abs(c_img)) > 0.1
    return {"p_base": float(p_base), "p_ts": float(p_ts), "p_img": float(p_img),
            "delta_ts": d_ts, "delta_img": d_img, "c_ts": c_ts, "c_img": c_img,
            "ts_capped": abs(raw_ts) > cap, "img_capped": abs(raw_img) > cap,
            "z": z, "composite": float(sigmoid(z)), "disagreement": disagree, "opposite_direction": bool(opposite)}
