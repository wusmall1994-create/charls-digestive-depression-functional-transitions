from pathlib import Path
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

SOURCE = ROOT / "outputs" / "comorbidity_state"
OUT = ROOT / "outputs" / "digestive_depression_multistate"
GRIP = ROOT / "outputs" / "grip_discovery" / "charls_grip_discovery_person_level.csv"
BASE = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + "
    "current_smoker + drank_last_year + binary_covariate_missing + baseline_bmi_imp + "
    "bmi_missing + log_hh_income_imp + income_missing + other_chronic_count"
)


def contrast(fit, exposure="state2013"):
    terms = {}
    for code in [2, 3]:
        terms[code] = next(t for t in [f"C({exposure}, Treatment(reference=0))[T.{code}]",
                                       f"C({exposure}, Treatment(reference=0))[T.{float(code)}]"] if t in fit.params.index)
    b = float(fit.params[terms[3]] - fit.params[terms[2]])
    c = fit.cov_params()
    se = float(np.sqrt(max(c.loc[terms[3], terms[3]] + c.loc[terms[2], terms[2]] - 2*c.loc[terms[3], terms[2]], 0)))
    return b, se


long = pd.read_csv(SOURCE / "comorbidity_state_long.csv", dtype={"ID": str})
intervals = pd.read_csv(SOURCE / "functional_multistate_intervals_weighted.csv", dtype={"ID": str})

# Sequence audit: only incident co-occurrence after wave 1 has an observable order.
seq_rows = []
for pid, g in long.sort_values(["ID", "wave"]).groupby("ID", sort=False):
    g = g[g[["digestive", "depression"]].notna().all(axis=1)]
    if g.empty:
        continue
    co = g[g.state.eq(3)]
    if co.empty or int(co.wave.iloc[0]) == 1:
        continue
    cw = int(co.wave.iloc[0])
    pre = g[g.wave.le(cw)]
    dw = pre.loc[pre.digestive.eq(1), "wave"]
    pw = pre.loc[pre.depression.eq(1), "wave"]
    if dw.empty or pw.empty:
        continue
    fd, fp = int(dw.min()), int(pw.min())
    sequence = "Digestive first" if fd < fp else "Depression first" if fp < fd else "Same observed wave"
    seq_rows.append({"ID": pid, "first_cooccurrence_wave": cw, "sequence": sequence})
seq = pd.DataFrame(seq_rows)
landmark = seq.merge(intervals, left_on=["ID", "first_cooccurrence_wave"], right_on=["ID", "wave"], how="left")
landmark["iadl_to_adl"] = ((landmark.functional_state.eq(1)) & landmark.destination_state.eq(2)).astype(int)
audit = landmark.groupby("sequence", observed=True).agg(
    incident_cooccurrence_people=("ID", "nunique"),
    observed_postformation_intervals=("destination_state", lambda s: int(s.notna().sum())),
    iadl_origin_intervals=("functional_state", lambda s: int(s.eq(1).sum())),
    iadl_to_adl_events=("iadl_to_adl", "sum"),
).reset_index()
audit["adjusted_model_feasible"] = (audit.iadl_origin_intervals >= 100) & (audit.iadl_to_adl_events >= 20)
audit.to_csv(OUT / "table_cooccurrence_sequence_feasibility.csv", index=False, encoding="utf-8-sig")
landmark[["ID", "first_cooccurrence_wave", "sequence", "functional_state", "destination_state"]].to_csv(
    OUT / "cooccurrence_sequence_landmark_data.csv", index=False, encoding="utf-8-sig")

# Grip pathway: 2013 exposure -> 2015 grip -> 2015-2018 IADL-stage transition.
g = pd.read_csv(GRIP, dtype={"person_id": str})[["person_id", "grip_2013", "grip_2015"]].rename(columns={"person_id": "ID"})
function_long = pd.read_csv(SOURCE / "functional_multistate_long.csv", dtype={"ID": str})
state13 = function_long[function_long.wave.eq(2)][["ID", "state", "functional_state"]].rename(columns={"state": "state2013", "functional_state": "function2013"})
data = intervals[intervals.wave.eq(3)].merge(state13, on="ID", how="inner", validate="many_to_one").merge(g, on="ID", how="left", validate="many_to_one")
for year in [2013, 2015]:
    v = f"grip_{year}"
    data[f"{v}_missing"] = data[v].isna().astype(int)
    data[f"{v}_imp"] = data[v].fillna(data[v].median())
data["grip2015_z"] = data.groupby("female")["grip_2015"].transform(lambda s: (s - s.mean()) / s.std())
data["grip2013_z"] = data.groupby("female")["grip_2013"].transform(lambda s: (s - s.mean()) / s.std())
data["grip2015_z_imp"] = data["grip2015_z"].fillna(0)

med = data[data.state2013.isin([2, 3])].drop_duplicates("ID").copy()
med["cooccurring2013"] = med.state2013.eq(3).astype(int)
med_complete = med.dropna(subset=["grip2015_z", "grip2013_z"]).copy()
med_formula = "grip2015_z ~ cooccurring2013 + grip2013_z + C(function2013) + " + BASE
med_ols = smf.ols(med_formula, med_complete)
med_fit = med_ols.fit(cov_type="HC3")
mb, mse = float(med_fit.params["cooccurring2013"]), float(med_fit.bse["cooccurring2013"])
mediator = pd.DataFrame([{
    "comparison": "2013 co-occurring vs depression only", "outcome": "Sex-standardized grip strength in 2015",
    "mean_difference_sd": mb, "ci_low": mb - 1.96*mse, "ci_high": mb + 1.96*mse,
    "p_value": float(2*(1-norm.cdf(abs(mb/mse)))), "n_people": int(med_fit.nobs)
}])
mediator.to_csv(OUT / "table_grip_exposure_link.csv", index=False, encoding="utf-8-sig")

iadl = data[data.functional_state.eq(1) & data.state2013.notna()].copy()
out_rows = []
for label, dest in [("IADL limitation to ADL limitation", [2]), ("IADL limitation to no limitation (recovery)", [0])]:
    iadl["event"] = iadl.destination_state.isin(dest).astype(int)
    base_formula = "event ~ C(state2013, Treatment(reference=0)) + " + BASE
    grip_formula = base_formula + " + grip2015_z_imp + grip_2015_missing"
    fits = []
    for formula in [base_formula, grip_formula]:
        model = smf.glm(formula, iadl, family=sm.families.Poisson(link=sm.families.links.Log()), freq_weights=iadl.iow)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.append(model.fit(cov_type="cluster", cov_kwds={"groups": iadl.loc[model.data.row_labels, "ID"]}))
    b0, se0 = contrast(fits[0]); b1, se1 = contrast(fits[1])
    out_rows.append({"transition": label, "primary_ratio": float(np.exp(b0)), "primary_ci_low": float(np.exp(b0-1.96*se0)),
                     "primary_ci_high": float(np.exp(b0+1.96*se0)), "grip_adjusted_ratio": float(np.exp(b1)),
                     "grip_adjusted_ci_low": float(np.exp(b1-1.96*se1)), "grip_adjusted_ci_high": float(np.exp(b1+1.96*se1)),
                     "absolute_log_ratio_attenuation_percent": float(100*(abs(b0)-abs(b1))/abs(b0)) if abs(b0)>1e-12 else np.nan,
                     "n_intervals": int(fits[1].nobs), "events": int(iadl.event.sum()), "grip2015_missing": int(iadl.grip_2015.isna().sum())})
grip_out = pd.DataFrame(out_rows)
grip_out.to_csv(OUT / "table_temporally_ordered_grip_pathway.csv", index=False, encoding="utf-8-sig")

qa = {
    "sequence": {"incident_cooccurrence_people": int(seq.ID.nunique()), "feasibility_rule": ">=100 IADL-origin intervals and >=20 IADL-to-ADL events in every sequence group",
                 "all_groups_feasible": bool(audit.adjusted_model_feasible.all())},
    "grip": {"mediator_complete_n": int(med_fit.nobs), "iadl_origin_intervals": int(len(iadl)),
             "grip2015_observed": int(iadl.grip_2015.notna().sum()), "temporal_order": "2013 state -> 2015 grip -> 2015-2018 transition",
             "estimand_boundary": "Exploratory temporally ordered attenuation; no natural indirect effect."}
}
(OUT / "sequence_and_grip_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))
print(audit.to_string(index=False))
print(mediator.to_string(index=False))
print(grip_out.to_string(index=False))

