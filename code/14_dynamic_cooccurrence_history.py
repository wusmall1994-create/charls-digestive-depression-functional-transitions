from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".analysis_deps"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


SOURCE = ROOT / "outputs" / "comorbidity_state"
OUT = ROOT / "outputs" / "digestive_depression_value_upgrade_2026-09-05"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + "
    "current_smoker + drank_last_year + baseline_bmi_imp + "
    "bmi_missing + log_hh_income_imp + income_missing + other_chronic_count + C(interval)"
)


def fit_modified_poisson(data: pd.DataFrame, formula: str, weight: str = "iow"):
    model = smf.glm(
        formula,
        data=data,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        freq_weights=data[weight],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(cov_type="cluster", cov_kwds={"groups": data.loc[model.data.row_labels, "ID"]})


def term_result(fit, term: str, label: str, transition: str, n: int, events: int) -> dict:
    b = float(fit.params[term])
    se = float(fit.bse[term])
    return {
        "transition": transition,
        "comparison": label,
        "risk_ratio": float(np.exp(b)),
        "ci_low": float(np.exp(b - 1.96 * se)),
        "ci_high": float(np.exp(b + 1.96 * se)),
        "p_value": float(fit.pvalues[term]),
        "n_intervals": int(n),
        "events": int(events),
    }


long = pd.read_csv(SOURCE / "comorbidity_state_long.csv", dtype={"ID": str})
intervals = pd.read_csv(SOURCE / "functional_multistate_intervals_weighted.csv", dtype={"ID": str})
intervals = intervals[intervals["iow"].notna()].copy()

# Construct exposure history using information observed no later than each interval origin.
hist_rows = []
for pid, g in long.sort_values(["ID", "wave"]).groupby("ID", sort=False):
    by_wave = g.set_index("wave")
    for origin_wave in (2, 3, 4):
        if origin_wave not in by_wave.index:
            continue
        history = by_wave.loc[[w for w in range(1, origin_wave + 1) if w in by_wave.index]].copy()
        expected = origin_wave
        complete = len(history) == expected and history["state"].notna().all()
        current = by_wave.loc[origin_wave]
        if isinstance(current, pd.DataFrame):
            current = current.iloc[0]
        co_count = int((history["state"] == 3).sum()) if complete else np.nan
        prior_co = int((history.iloc[:-1]["state"] == 3).any()) if complete else np.nan
        current_co = bool(current["state"] == 3) if pd.notna(current["state"]) else False
        if not complete:
            history_group = "Incomplete exposure history"
        elif current["state"] == 2 and prior_co == 0:
            history_group = "Current depressive symptoms only without prior co-occurrence"
        elif current["state"] == 2 and prior_co == 1:
            history_group = "Current depressive symptoms only after prior co-occurrence"
        elif current_co and prior_co == 0:
            history_group = "First observed co-occurrence"
        elif current_co and prior_co == 1:
            history_group = "Repeated co-occurrence"
        else:
            history_group = "Other current state"
        hist_rows.append(
            {
                "ID": pid,
                "wave": origin_wave,
                "complete_exposure_history": int(complete),
                "prior_cooccurrence": prior_co,
                "cooccurrence_count_to_origin": co_count,
                "cooccurrence_history_group": history_group,
            }
        )

history = pd.DataFrame(hist_rows)
data = intervals.merge(history, on=["ID", "wave"], how="left", validate="many_to_one")
data.to_csv(OUT / "dynamic_exposure_interval_data.csv", index=False, encoding="utf-8-sig")

# Primary history contrast: among current depressive-symptom intervals, distinguish first and repeated
# co-occurrence and a return to depressive symptoms only after earlier co-occurrence.
analysis = data[
    data["complete_exposure_history"].eq(1)
    & data["cooccurrence_history_group"].isin(
        ["Current depressive symptoms only without prior co-occurrence", "Current depressive symptoms only after prior co-occurrence", "First observed co-occurrence", "Repeated co-occurrence"]
    )
].copy()
analysis["history_group"] = pd.Categorical(
    analysis["cooccurrence_history_group"],
    categories=["Current depressive symptoms only without prior co-occurrence", "Current depressive symptoms only after prior co-occurrence", "First observed co-occurrence", "Repeated co-occurrence"],
)

definitions = [
    (1, 2, "IADL limitation to ADL limitation"),
    (1, 0, "IADL limitation to no limitation (recovery)"),
    (0, 2, "No limitation to ADL limitation"),
    (2, [0, 1], "ADL limitation to functional improvement"),
]
model_rows = []
count_rows = []
for origin, destination, label in definitions:
    x = analysis[analysis["functional_state"].eq(origin)].copy()
    destinations = destination if isinstance(destination, list) else [destination]
    x["event"] = x["destination_state"].isin(destinations).astype(int)
    summary = x.groupby("history_group", observed=False).agg(
        intervals=("ID", "size"), people=("ID", "nunique"), events=("event", "sum")
    ).reset_index()
    summary.insert(0, "transition", label)
    count_rows.extend(summary.to_dict("records"))
    fit = fit_modified_poisson(
        x,
        "event ~ C(history_group, Treatment(reference='Current depressive symptoms only without prior co-occurrence')) + " + BASE,
    )
    for group in ["Current depressive symptoms only after prior co-occurrence", "First observed co-occurrence", "Repeated co-occurrence"]:
        term = f"C(history_group, Treatment(reference='Current depressive symptoms only without prior co-occurrence'))[T.{group}]"
        model_rows.append(term_result(fit, term, f"{group} vs current depressive symptoms only without prior co-occurrence", label, len(x), int(x.event.sum())))

pd.DataFrame(count_rows).to_csv(OUT / "table_dynamic_history_event_counts.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(model_rows).to_csv(OUT / "table_dynamic_history_transition_models.csv", index=False, encoding="utf-8-sig")

# Dose-response among current co-occurrence intervals: one prior/current episode versus at least two.
dose = data[
    data["complete_exposure_history"].eq(1) & data["state"].eq(3) & data["cooccurrence_count_to_origin"].ge(1)
].copy()
dose["cooccurrence_burden"] = np.where(
    dose["cooccurrence_count_to_origin"].eq(1), "One observed wave", "At least two observed waves"
)
dose["cooccurrence_burden"] = pd.Categorical(
    dose["cooccurrence_burden"], categories=["One observed wave", "At least two observed waves"]
)
dose_rows = []
dose_counts = []
for origin, destination, label in definitions[:2]:
    x = dose[dose["functional_state"].eq(origin)].copy()
    x["event"] = x["destination_state"].eq(destination).astype(int)
    s = x.groupby("cooccurrence_burden", observed=False).agg(
        intervals=("ID", "size"), people=("ID", "nunique"), events=("event", "sum")
    ).reset_index()
    s.insert(0, "transition", label)
    dose_counts.extend(s.to_dict("records"))
    fit = fit_modified_poisson(
        x,
        "event ~ C(cooccurrence_burden, Treatment(reference='One observed wave')) + " + BASE,
    )
    term = "C(cooccurrence_burden, Treatment(reference='One observed wave'))[T.At least two observed waves]"
    dose_rows.append(term_result(fit, term, "At least two vs one observed co-occurrence waves", label, len(x), int(x.event.sum())))

pd.DataFrame(dose_counts).to_csv(OUT / "table_cumulative_burden_event_counts.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(dose_rows).to_csv(OUT / "table_cumulative_burden_transition_models.csv", index=False, encoding="utf-8-sig")

# Incident co-occurrence sequence: descriptive longitudinal follow-up from the first observed co-occurrence.
sequence_rows = []
for pid, g in long.sort_values(["ID", "wave"]).groupby("ID", sort=False):
    g = g[g[["digestive", "depression"]].notna().all(axis=1)].copy()
    co = g[g["state"].eq(3)]
    if co.empty or int(co.iloc[0]["wave"]) == 1:
        continue
    first_co_wave = int(co.iloc[0]["wave"])
    before = g[g["wave"].le(first_co_wave)]
    first_d = int(before.loc[before["digestive"].eq(1), "wave"].min())
    first_p = int(before.loc[before["depression"].eq(1), "wave"].min())
    sequence = "Digestive disease first" if first_d < first_p else "Depressive symptoms first" if first_p < first_d else "Same observed wave"
    sequence_rows.append({"ID": pid, "first_cooccurrence_wave": first_co_wave, "sequence": sequence})

sequence = pd.DataFrame(sequence_rows)
seq_data = data.merge(sequence, on="ID", how="inner")
seq_data = seq_data[seq_data["wave"].ge(seq_data["first_cooccurrence_wave"]) & seq_data["functional_state"].eq(1)].copy()
seq_data["progression"] = seq_data["destination_state"].eq(2).astype(int)
seq_data["recovery"] = seq_data["destination_state"].eq(0).astype(int)
seq_summary = seq_data.groupby("sequence", observed=True).agg(
    people=("ID", "nunique"), iadl_origin_intervals=("ID", "size"),
    iadl_to_adl_events=("progression", "sum"), recovery_events=("recovery", "sum")
).reset_index()
seq_summary["progression_percent"] = 100 * seq_summary["iadl_to_adl_events"] / seq_summary["iadl_origin_intervals"]
seq_summary["recovery_percent"] = 100 * seq_summary["recovery_events"] / seq_summary["iadl_origin_intervals"]
seq_summary.to_csv(OUT / "table_sequence_longitudinal_descriptive.csv", index=False, encoding="utf-8-sig")

qa = {
    "source_intervals": int(len(intervals)),
    "people": int(intervals["ID"].nunique()),
    "complete_history_intervals": int(data["complete_exposure_history"].eq(1).sum()),
    "dynamic_history_analysis_intervals": int(len(analysis)),
    "current_cooccurrence_dose_intervals": int(len(dose)),
    "incident_sequence_people": int(sequence["ID"].nunique()),
    "sequence_iadl_intervals": int(len(seq_data)),
    "estimand_boundary": (
        "History analyses are transition-specific adjusted associations. Exposure history uses only waves "
        "observed through each interval origin; it is not a clinical severity measure or a causal dose-response."
    ),
}
(OUT / "dynamic_upgrade_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))
print("\nDYNAMIC HISTORY MODELS")
print(pd.DataFrame(model_rows).to_string(index=False))
print("\nCUMULATIVE BURDEN MODELS")
print(pd.DataFrame(dose_rows).to_string(index=False))
print("\nSEQUENCE DESCRIPTIVE")
print(seq_summary.to_string(index=False))


