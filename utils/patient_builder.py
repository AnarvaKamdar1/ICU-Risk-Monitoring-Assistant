"""Builds the simulated composite patient from the three independent test sets."""
import numpy as np
import pandas as pd

from timeseries_ext import featurize_long


def series_48h(ts, raw_all, idx):
    """p_ts for hours warmup..48 for many 48-h records at once."""
    import torch
    from ts_inference import featurize, make_batch
    W, start = ts.cfg["window"], ts.cfg["warmup"]
    X = torch.tensor(featurize(raw_all[idx], ts.stats), device=ts.device)
    ends = np.arange(start, 49)
    pid = np.repeat(np.arange(len(idx)), len(ends)); end = np.tile(ends, len(idx))
    out = []
    with torch.no_grad():
        for s in range(0, len(pid), 4096):
            xb, pad = make_batch(X, torch.tensor(pid[s:s + 4096], device=ts.device),
                                 torch.tensor(end[s:s + 4096], device=ts.device), W)
            out.append(ts.model(xb, pad).float().cpu().numpy())
    return ts.calibrate(np.concatenate(out)).reshape(len(idx), len(ends)), ends


def pick_first_segment(ts, npz, scenario, trig_cfg, rng, fixed=None):
    from timeseries_ext import TriggerMonitor
    rids, raw_all, labels = npz["record_ids"], npz["raw"], npz["labels"]
    if fixed is not None:
        return int(np.where(rids == fixed)[0][0])
    pool = np.where(labels == (1 if scenario == "deteriorating" else 0))[0]
    P, ends = series_48h(ts, raw_all, pool)
    good = []
    for k, traj in enumerate(P):
        mon = TriggerMonitor(trig_cfg, int(ends[0]))
        n_trig = sum(mon.step(h, p) is not None for h, p in zip(ends, traj))
        rising = traj[-6:].mean() - traj[:6].mean()
        if scenario == "deteriorating" and n_trig >= 1 and rising > 0.02:
            good.append(k)
        if scenario == "improving" and rising < -0.02:
            good.append(k)
    cand = pool[good] if good else pool
    return int(rng.choice(cand))


def chain_segments(ts, npz, first_idx, scenario, max_hours, rng):
    """Concatenate 48-h records until max_hours is covered. Each next record is the one (from the same
    outcome group) whose first 12 h look most like the last 12 h of the chain - a simulation device."""
    rids, raw_all, labels = npz["record_ids"], npz["raw"], npz["labels"]
    W, V = ts.cfg["window"], raw_all.shape[2]
    pool = list(np.where(labels == (1 if scenario == "deteriorating" else 0))[0])
    used, chain = [first_idx], [raw_all[first_idx]]
    segments = [{"record_id": int(rids[first_idx]), "outcome_died": int(labels[first_idx]), "start_hour": 0}]
    while sum(len(c) for c in chain) < max_hours + 1:
        cur = np.concatenate(chain)
        sig = featurize_long(cur, ts.stats, W)[0, -W:, :V].mean(0)
        best, best_d = None, np.inf
        for k in pool:
            if k in used:
                continue
            cand = featurize_long(raw_all[k], ts.stats, W)[0, :W, :V].mean(0)
            d = float(np.mean((cand - sig) ** 2))
            if d < best_d:
                best, best_d = k, d
        if best is None:
            break
        segments.append({"record_id": int(rids[best]), "outcome_died": int(labels[best]),
                         "start_hour": int(len(cur)), "match_distance": round(best_d, 4)})
        used.append(best); chain.append(raw_all[best])
    return np.concatenate(chain), pd.DataFrame(segments)


def pick_wids(test_df, age, sex, rng, fixed=None, prefer=("Respiratory", "Sepsis")):
    if fixed is not None:
        return test_df[test_df.encounter_id == fixed].iloc[0]
    d = test_df.copy()
    for tol, use_dx in [(5, True), (5, False), (10, False), (200, False)]:
        m = (d.gender == sex) & ((d.age - age).abs() <= tol)
        if use_dx:
            m &= d.apache_3j_bodysystem.isin(prefer)
        if m.any():
            return d[m].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
    return d.sample(1, random_state=int(rng.integers(1e9))).iloc[0]


def pick_image(pool, tier, age, sex, rng, fixed=None, exclude=()):
    if fixed is not None:
        return pool[pool.patientId == fixed].iloc[0]
    d = pool[~pool.patientId.isin(exclude)]
    for tol in (10, 20, 200):
        m = (d.tier == tier) & (d.sex == sex) & ((d.age - age).abs() <= tol)
        if m.any():
            return d[m].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
    sub = d[d.tier == tier] if (d.tier == tier).any() else d
    return sub.sample(1, random_state=int(rng.integers(1e9))).iloc[0]


def next_image(pool, prev_sev, prev_p, scenario, rng, exclude=()):
    """Simulated follow-up: the smallest step up (deteriorating) or down (improving) in stored severity."""
    d = pool[~pool.patientId.isin(exclude)].copy()
    if d.empty:                                   # pool exhausted -> allow re-use of earlier images (except the current one)
        d = pool[pool.patientId != list(exclude)[-1]].copy() if len(exclude) else pool.copy()
    if scenario == "deteriorating":
        c = d[(d.brixia_total > prev_sev) | ((d.brixia_total == prev_sev) & (d.p_img > prev_p + 0.05))]
        if len(c):
            step = c.brixia_total.min()
            return c[c.brixia_total == step].sample(1, random_state=int(rng.integers(1e9))).iloc[0], "worse"
        return d.sort_values(["brixia_total", "p_img"]).iloc[-1], "worst available"
    c = d[(d.brixia_total < prev_sev) | ((d.brixia_total == prev_sev) & (d.p_img < prev_p - 0.05))]
    if len(c):
        step = c.brixia_total.max()
        return c[c.brixia_total == step].sample(1, random_state=int(rng.integers(1e9))).iloc[0], "better"
    return d.sort_values(["brixia_total", "p_img"]).iloc[0], "best available"
