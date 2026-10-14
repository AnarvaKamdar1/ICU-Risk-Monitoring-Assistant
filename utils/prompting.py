"""Structured JSON summary, grounded prompt, template summary and a numeric-grounding check."""
import json
import numbers
import re

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant inside a research prototype. You summarise the output of three "
    "risk models for clinicians. Rules:\n"
    "1. Use ONLY facts and numbers present in the JSON. Do not invent findings, diagnoses, medications or numbers.\n"
    "2. The composite index is a relative risk index built from independent models. It is NOT a probability of death; "
    "never call it one.\n"
    "3. The reference notes describe OTHER patients. Use them only as general clinical context and say so; never "
    "attribute their content to this patient.\n"
    "4. Re-imaging is decided by a fixed monitoring rule, not by you. Report it, do not recommend treatment.\n"
    "5. Write plain clinical English, at most 180 words, in exactly these sections:\n"
    "OVERALL STATUS / KEY DRIVERS / CHANGE SINCE LAST REPORT / AGREEMENT BETWEEN MODELS / CONTEXT FROM REFERENCE NOTES / LIMITATIONS"
)


def r(x, n=3):
    return None if x is None else round(float(x), n)


def build_summary(rep):
    """rep: dict assembled by the monitor for one report."""
    f = rep["fusion"]
    return {
        "report": {"number": rep["number"], "type": rep["type"], "icu_hour": rep["hour"],
                   "hours_since_previous_image": rep.get("hours_since_prev_image"),
                   "monitoring_trigger": rep.get("trigger_reason")},
        "patient_background": rep["background"],
        "baseline_model": {"p_base": r(f["p_base"]),
                           "top_factors": [{"feature": t["feature"], "value": str(t["value"]),
                                            "effect": "raises" if t["shap"] > 0 else "lowers"} for t in rep["shap_top"]]},
        "vitals_model": {"p_ts": r(f["p_ts"]), "p_ts_at_previous_report": r(rep.get("prev_p_ts")),
                         "top_variables": [{"variable": t["variable"], "last_value": r(t["last"], 2), "trend": t["trend"],
                                            "effect": "raises" if t["contribution"] > 0 else "lowers"} for t in rep["ts_top"]]},
        "imaging_model": {"p_img": r(f["p_img"]), "severity_total_0_18": rep["image"]["brixia_total"],
                          "severity_band": rep["image"]["band"], "bilateral": rep["image"]["bilateral"],
                          "zones_involved": [z for z, g in rep["image"]["zone_grades"].items() if g > 0],
                          "n_boxes": rep["image"]["n_boxes"], "previous_severity_total": rep.get("prev_severity"),
                          "severity_change": rep.get("severity_change")},
        "fusion": {"composite_index": r(f["composite"]), "composite_at_previous_report": r(rep.get("prev_composite")),
                   "vitals_contribution_logodds": r(f["c_ts"]), "imaging_contribution_logodds": r(f["c_img"]),
                   "alpha": r(rep["alpha"]), "beta": r(rep["beta"]), "disagreement": r(f["disagreement"]),
                   "models_point_in_opposite_directions": f["opposite_direction"],
                   "composite_since_previous_image": rep.get("since_last_image")},
        "reference_notes": [{"specialty": n["specialty"], "title": n["title"], "excerpt": n["excerpt"]} for n in rep["notes"]],
        "caveats": ["composite patient simulated from three unrelated datasets",
                    "follow-up X-rays are simulated (images from other patients)",
                    "composite index is not a calibrated probability of death"],
    }


def build_prompt(summary):
    return ("Summarise this monitoring report for the clinical team.\n\nDATA (JSON):\n" +
            json.dumps(summary, indent=1, default=str))


def template_summary(s):
    f, b, v, i = s["fusion"], s["baseline_model"], s["vitals_model"], s["imaging_model"]
    drivers = ", ".join(f"{t['feature']} ({t['effect']} risk)" for t in b["top_factors"][:3])
    vit = ", ".join(f"{t['variable']} {t['trend'] or 'single value'}" for t in v["top_variables"][:3])
    chg = ("first report" if f["composite_at_previous_report"] is None else
           f"composite {f['composite_at_previous_report']} -> {f['composite_index']}; imaging severity "
           f"{i['previous_severity_total']} -> {i['severity_total_0_18']} ({i['severity_change']})")
    agree = ("vitals and imaging point in opposite directions" if f["models_point_in_opposite_directions"]
             else "vitals and imaging point in the same direction") + f" (disagreement {f['disagreement']} log-odds)"
    return (f"OVERALL STATUS: composite risk index {f['composite_index']} (relative index, not a probability of death).\n"
            f"KEY DRIVERS: baseline p_base {b['p_base']} driven by {drivers}; vitals p_ts {v['p_ts']} ({vit}); "
            f"imaging severity {i['severity_total_0_18']}/18 ({i['severity_band']}), p_img {i['p_img']}.\n"
            f"CHANGE SINCE LAST REPORT: {chg}.\n"
            f"AGREEMENT BETWEEN MODELS: {agree}.\n"
            f"CONTEXT FROM REFERENCE NOTES: {len(s['reference_notes'])} notes from other patients retrieved as general context.\n"
            f"LIMITATIONS: " + "; ".join(s["caveats"]) + ".")


def _numbers(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, out)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, numbers.Number):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in re.findall(r"-?\d+(?:\.\d+)?", obj):
            out.add(float(m))
    return out


def ungrounded_numbers(text, summary, ignore_below=10):
    """Numbers in the LLM text that do not match any number in the JSON (allowing rounding and % form)."""
    allowed = _numbers(summary, set())
    variants = set()
    for a in allowed:
        for x in (a, a * 100):
            for d in (0, 1, 2, 3):
                variants.add(round(x, d))
    bad = []
    for m in re.findall(r"-?\d+(?:\.\d+)?", text):
        x = float(m)
        if abs(x) < ignore_below and float(x).is_integer():
            continue
        if not any(abs(x - v) <= 0.0051 for v in variants):
            bad.append(m)
    return sorted(set(bad))
