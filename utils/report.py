"""Printed report: patient data, SHAP bars, trend chart, X-ray(s), reference notes, summary."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

COLORS = {"base": "#7f8c8d", "ts": "#2e86c1", "img": "#e67e22", "comp": "#c0392b"}


def trend_chart(hist, events, segments, current_hour, last_image_hour, theta, critical, ax_top, ax_bot):
    h = hist["hour"].values
    ax_top.plot(h, hist["p_base"], color=COLORS["base"], ls="--", lw=1.4, label="baseline p_base (once)")
    ax_top.step(h, hist["p_img"], where="post", color=COLORS["img"], lw=1.6, label="imaging p_img (per image)")
    ax_top.plot(h, hist["p_ts"], color=COLORS["ts"], lw=1.0, alpha=0.8, label="vitals p_ts (hourly)")
    ax_top.plot(h, hist["composite"], color=COLORS["comp"], lw=2.4, label="composite index")
    ax_top.axhline(theta, color=COLORS["ts"], ls=":", lw=0.9, label=f"re-imaging θ for p_ts ({theta:.2f})")
    ax_top.axhline(critical, color=COLORS["comp"], ls=":", lw=0.9, label=f"critical composite ({critical:.2f})")
    ax_top.axvspan(last_image_hour, current_hour, color=COLORS["comp"], alpha=0.07, label="since last image")
    for e in events:
        ax_top.axvline(e["hour"], color="k", lw=0.8)
        ax_top.text(e["hour"] + 0.3, 0.97, f"img {e['image_no']}", rotation=90, fontsize=7, va="top", ha="left",
                    transform=ax_top.get_xaxis_transform())
    for s in segments[1:]:
        ax_top.axvline(s, color="grey", lw=0.6, ls=(0, (1, 3)))
    ax_top.set(ylim=(0, 1), ylabel="probability / index", xlim=(h.min(), max(h.max(), current_hour)))
    ax_top.legend(fontsize=7, loc="upper left", ncol=2)
    ax_bot.axhline(0, color="k", lw=0.6)
    ax_bot.plot(h, hist["c_ts"], color=COLORS["ts"], label="α·Δ_ts (vitals)")
    ax_bot.step(h, hist["c_img"], where="post", color=COLORS["img"], label="β·Δ_img (imaging)")
    ax_bot.fill_between(h, 0, hist["c_ts"] + hist["c_img"], color=COLORS["comp"], alpha=0.12, label="total adjustment")
    ax_bot.set(ylabel="log-odds", xlabel="ICU hour (dotted grey = next data segment, simulation)")
    ax_bot.legend(fontsize=7, loc="upper left", ncol=3)


def shap_bars(shap_top, p_base, ax):
    t = shap_top.iloc[::-1]
    fmt = lambda v: f"{v:.3g}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else str(v)
    ax.barh([f"{f} = {fmt(v)}" for f, v in zip(t.feature, t.value)], t.shap,
            color=["#c0392b" if s > 0 else "#2e86c1" for s in t.shap])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"Baseline (once): p_base = {p_base:.3f}\nSHAP, red raises / blue lowers", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)


def render(rep, hist, events, segments, im, theta, critical, prev_assessment=None):
    f = rep["fusion"]
    hdr = ("Initial assessment (t = 0)" if rep["type"] == "initial"
           else f"Re-imaging #{rep['image_no'] - 1} — trigger: {rep['trigger_reason']}")
    display(Markdown(f"---\n## REPORT {rep['number']} · ICU hour {rep['hour']} · {hdr}"))

    bg = rep["background"]
    vit = rep["latest_vitals"]
    pt = pd.DataFrame({"": ["age / sex", "BMI", "diagnosis group", "admission", "comorbidities", "source records"],
                       "value": [f"{bg['age']} / {bg['sex']}", bg["bmi"], bg["diagnosis_group"], bg["admission"],
                                 ", ".join(bg["comorbidities"]) or "none recorded", rep["source_ids"]]})
    display(Markdown("**Patient data** (simulated composite patient)"))
    display(pt.set_index(""))
    display(pd.DataFrame([{k: (f"{v[0]:.1f}" if not np.isnan(v[0]) else "—") + (f" ({v[1]} h ago)" if v[1] else "")
                           for k, v in vit.items()}], index=["latest vitals/labs"]))

    scores = pd.DataFrame({
        "model": ["baseline (XGBoost)", "vitals (Transformer)", "imaging (Faster R-CNN)", "COMPOSITE INDEX"],
        "probability": [f["p_base"], f["p_ts"], f["p_img"], f["composite"]],
        "contribution (log-odds)": [np.log(f["p_base"] / (1 - f["p_base"])), f["c_ts"], f["c_img"], f["z"]],
        "note": ["anchor", f"α={rep['alpha']:.2f}" + (" (capped)" if f["ts_capped"] else ""),
                 f"β={rep['beta']:.2f}" + (" (capped)" if f["img_capped"] else ""),
                 f"disagreement {f['disagreement']:.2f}" + (" — opposite directions" if f["opposite_direction"] else "")]})
    display(Markdown(f"**Scores** — hours since previous image: {rep.get('hours_since_prev_image') or 0}"))
    display(scores.set_index("model").round(3))
    if rep.get("since_last_image"):
        s = rep["since_last_image"]
        display(Markdown(f"Composite since previous image (hour {s['from_hour']}→{rep['hour']}): "
                         f"{s['start']:.3f} → {s['end']:.3f}, mean {s['mean']:.3f}, peak {s['peak']:.3f}"))

    n_img = 2 if prev_assessment is not None else 1
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 1 + n_img, width_ratios=[1.3] + [1] * n_img)
    shap_bars(rep["shap_df"], f["p_base"], fig.add_subplot(gs[0]))
    if prev_assessment is not None:
        im.draw(prev_assessment, ax=fig.add_subplot(gs[1]),
                title=f"previous image: severity {prev_assessment['brixia_total']}/18 ({prev_assessment['band']})")
    a = rep["assessment"]
    im.draw(a, ax=fig.add_subplot(gs[-1]),
            title=f"current image: severity {a['brixia_total']}/18 ({a['band']}), p_img={a['p_img']:.2f}"
                  + (f"\n{rep['severity_change']} vs previous" if rep.get("severity_change") else ""))
    plt.tight_layout(); plt.savefig(rep["fig_prefix"] + "_panel.png", dpi=120); plt.show()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    trend_chart(hist, events, segments, rep["hour"], rep["shade_from"], theta, critical, a1, a2)
    a1.set_title(f"Trend up to ICU hour {rep['hour']}", fontsize=10)
    plt.tight_layout(); plt.savefig(rep["fig_prefix"] + "_trend.png", dpi=120); plt.show()

    display(Markdown("**Vitals model — variables driving the current window**"))
    display(rep["ts_df"][["variable", "contribution", "first", "last", "trend", "n_obs"]].round(3))

    display(Markdown("**Reference notes** (other patients, MTSamples — general context only)"))
    for n in rep["notes"]:
        display(Markdown(f"- *{n['specialty']} — {n['title']}* (similarity {n['score']}): “{n['excerpt']}”"))

    display(Markdown(f"**Summary** ({rep['summary_source']})"))
    print(rep["summary_text"])
    if rep["ungrounded"]:
        display(Markdown(f"⚠️ numbers in the summary not found in the input JSON: `{rep['ungrounded']}`"))
    else:
        display(Markdown("✓ every number in the summary appears in the input JSON"))
