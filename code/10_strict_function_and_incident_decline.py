from pathlib import Path
import os
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

DATA = Path(os.environ["CHARLS_DATA_DIR"]).expanduser().resolve()
SOURCE = ROOT / "outputs" / "comorbidity_state"
OUT = ROOT / "outputs" / "digestive_depression_multistate"
HARMONIZED = DATA / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta"
HSF2020 = DATA / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta"
SAMPLE2020 = DATA / "2020" / "CHARLS2020r" / "Sample_Infor.dta"

WAVE_YEAR = {2: 2013, 3: 2015, 4: 2018, 5: 2020}
FUNCTION_LABELS = {0: "No limitation", 1: "IADL limitation only", 2: "ADL limitation", 3: "Death"}
STATE_LABELS = {0: "Neither", 1: "Digestive only", 2: "Depression only", 3: "Co-occurring"}
KEY = [
    (0, [2], "No limitation to ADL limitation"),
    (1, [2], "IADL limitation to ADL limitation"),
    (1, [0], "IADL limitation to no limitation (recovery)"),
    (2, [0, 1], "ADL limitation to functional improvement"),
]
BASE = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + "
    "current_smoker + drank_last_year + binary_covariate_missing + baseline_bmi_imp + "
    "bmi_missing + log_hh_income_imp + income_missing + other_chronic_count"
)


def prepare(x):
    x = x.copy()
    x["baseline_age_c"] = x["baseline_age"] - 60
    x["education_cat"] = x["education_code"].fillna(-1).astype(str)
    x["marital_cat"] = x["marital_code"].fillna(-1).astype(str)
    x["binary_covariate_missing"] = x[["rural_hukou", "current_smoker", "drank_last_year"]].isna().any(axis=1).astype(int)
    for v in ["female", "rural_hukou", "current_smoker", "drank_last_year"]:
        x[v] = x[v].fillna(x[v].mode(dropna=True).iloc[0])
    x["bmi_missing"] = x["baseline_bmi"].isna().astype(int)
    x["baseline_bmi_imp"] = x["baseline_bmi"].fillna(x["baseline_bmi"].median())
    x["income_missing"] = x["log_hh_income"].isna().astype(int)
    x["log_hh_income_imp"] = x["log_hh_income"].fillna(x["log_hh_income"].median())
    return x


def observation_weights(x):
    den = smf.glm("outcome_observed ~ C(state) + C(functional_state) + C(interval) + " + BASE,
                  x, family=sm.families.Binomial()).fit()
    num = smf.glm("outcome_observed ~ C(functional_state) + C(interval)",
                  x, family=sm.families.Binomial()).fit()
    x = x.copy()
    x["iow_raw"] = np.clip(num.predict(x), .02, .995) / np.clip(den.predict(x), .02, .995)
    obs = x[x.outcome_observed.eq(1)].copy()
    lo, hi = obs.iow_raw.quantile([.01, .99])
    obs["iow"] = obs.iow_raw.clip(lo, hi)
    return obs, {"denominator_converged": bool(den.converged), "numerator_converged": bool(num.converged),
                 "p01": float(lo), "p99": float(hi)}


def term(fit, code):
    for t in [f"C(state, Treatment(reference=0))[T.{code}]", f"C(state, Treatment(reference=0))[T.{float(code)}]"]:
        if t in fit.params.index:
            return t
    raise KeyError(code)


def fit_model(risk, transition, destinations):
    risk = risk.copy()
    risk["event"] = risk.destination_state.isin(destinations).astype(int)
    formula = "event ~ C(state, Treatment(reference=0)) + C(interval) + " + BASE
    model = smf.glm(formula, risk, family=sm.families.Poisson(link=sm.families.links.Log()), freq_weights=risk.iow)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": risk.loc[model.data.row_labels, "ID"]})
    t3, t2 = term(fit, 3), term(fit, 2)
    logrr = float(fit.params[t3] - fit.params[t2])
    cov = fit.cov_params()
    se = float(np.sqrt(max(cov.loc[t3, t3] + cov.loc[t2, t2] - 2 * cov.loc[t3, t2], 0)))
    return {
        "transition": transition, "comparison": "Co-occurring vs depression only",
        "ratio": float(np.exp(logrr)), "ci_low": float(np.exp(logrr - 1.96 * se)),
        "ci_high": float(np.exp(logrr + 1.96 * se)),
        "p_value": float(2 * (1 - norm.cdf(abs(logrr / se)))) if se else np.nan,
        "n_intervals": int(fit.nobs), "events": int(risk.loc[model.data.row_labels, "event"].sum()),
        "people": int(risk.loc[model.data.row_labels, "ID"].nunique()), "converged": bool(fit.converged),
    }


long = pd.read_csv(SOURCE / "comorbidity_state_long.csv", dtype={"ID": str})
base_ids = set(long.loc[long.wave.eq(1), "ID"])
fixed_cols = ["ID", "communityID", "baseline_age", "female", "education_code", "marital_code",
              "rural_hukou", "current_smoker", "drank_last_year", "baseline_bmi", "log_hh_income", "other_chronic_count"]
fixed = long.loc[long.wave.eq(1), fixed_cols].drop_duplicates("ID")

hcols = ["ID"]
for w in [2, 3, 4]:
    hcols += [f"inw{w}", f"r{w}iwstat", f"r{w}adlab_c", f"r{w}adlabm_c", f"r{w}iadlza", f"r{w}iadlzam"]
h = pd.read_stata(HARMONIZED, columns=hcols, convert_categoricals=False)
h = h[h.ID.astype(str).isin(base_ids)].copy()
h["ID"] = h.ID.astype(str)
frames = []
for w in [2, 3, 4]:
    adl_count = h[f"r{w}adlab_c"].where(h[f"r{w}adlabm_c"].eq(0))
    iadl_count = h[f"r{w}iadlza"].where(h[f"r{w}iadlzam"].eq(0))
    ok = h[f"inw{w}"].eq(1) & adl_count.notna() & iadl_count.notna()
    fs = pd.Series(np.nan, index=h.index)
    fs.loc[ok & adl_count.lt(2) & iadl_count.lt(2)] = 0
    fs.loc[ok & adl_count.lt(2) & iadl_count.ge(2)] = 1
    fs.loc[ok & adl_count.ge(2)] = 2
    frames.append(pd.DataFrame({"ID": h.ID, "wave": w, "year": WAVE_YEAR[w], "alive_interview": h[f"inw{w}"].eq(1).astype(int),
                                "adl_difficulty_count": adl_count, "iadl_difficulty_count": iadl_count, "functional_state": fs}))

adl20_items = ["db001", "db003", "db005", "db007", "db009", "db011"]
iadl20_items = ["db014", "db016", "db018", "db020", "db022"]
hsf = pd.read_stata(HSF2020, columns=["ID"] + adl20_items + iadl20_items, convert_categoricals=False)
s20 = pd.read_stata(SAMPLE2020, convert_categoricals=False)
x20 = s20.merge(hsf, on="ID", how="left")
x20["ID"] = x20.ID.astype(str)
x20 = x20[x20.ID.isin(base_ids)].copy()
adl_ok = x20[adl20_items].isin([1, 2, 3, 4]).all(axis=1)
iadl_ok = x20[iadl20_items].isin([1, 2, 3, 4]).all(axis=1)
adl_count = x20[adl20_items].gt(1).sum(axis=1).astype(float).where(adl_ok)
iadl_count = x20[iadl20_items].gt(1).sum(axis=1).astype(float).where(iadl_ok)
ok = x20.died.eq(0) & adl_count.notna() & iadl_count.notna()
fs = pd.Series(np.nan, index=x20.index)
fs.loc[ok & adl_count.lt(2) & iadl_count.lt(2)] = 0
fs.loc[ok & adl_count.lt(2) & iadl_count.ge(2)] = 1
fs.loc[ok & adl_count.ge(2)] = 2
frames.append(pd.DataFrame({"ID": x20.ID, "wave": 5, "year": 2020, "alive_interview": x20.died.eq(0).astype(int),
                            "adl_difficulty_count": adl_count, "iadl_difficulty_count": iadl_count, "functional_state": fs}))

functional = pd.concat(frames, ignore_index=True)
exposure = long[["ID", "wave", "state", "digestive", "depression", "cesd10", "digestive_dispute"]]
functional = functional.merge(exposure, on=["ID", "wave"], how="left").merge(fixed, on="ID", how="left")
functional.to_csv(OUT / "strict_functional_multistate_long.csv", index=False, encoding="utf-8-sig")

rows = []
for sw, ew in [(2, 3), (3, 4), (4, 5)]:
    start = functional[functional.wave.eq(sw)].copy()
    end = functional[functional.wave.eq(ew)].set_index("ID")
    if ew in [3, 4]:
        vital = h.set_index("ID")[f"r{ew}iwstat"]
        dead, alive = start.ID.map(vital).eq(5), start.ID.map(vital).isin([1, 4])
    else:
        vital = s20.assign(ID=s20.ID.astype(str)).set_index("ID").died
        dead, alive = start.ID.map(vital).eq(1), start.ID.map(vital).eq(0)
    dest_living = start.ID.map(end.functional_state)
    dest = pd.Series(np.nan, index=start.index)
    dest.loc[dead] = 3
    dest.loc[alive & dest_living.notna()] = dest_living.loc[alive & dest_living.notna()]
    start["destination_state"] = dest
    start["outcome_observed"] = dest.notna().astype(int)
    start["interval"] = f"{WAVE_YEAR[sw]}-{WAVE_YEAR[ew]}"
    start["interval_years"] = WAVE_YEAR[ew] - WAVE_YEAR[sw]
    rows.append(start)
all_strict = pd.concat(rows, ignore_index=True)
all_strict = all_strict[all_strict.alive_interview.eq(1) & all_strict.functional_state.notna() & all_strict.state.notna()].copy()
all_strict = prepare(all_strict)
strict_obs, wqa = observation_weights(all_strict)
strict_obs.to_csv(OUT / "strict_functional_multistate_intervals_weighted.csv", index=False, encoding="utf-8-sig")

strict_results = []
for origin, destinations, label in KEY:
    strict_results.append(fit_model(strict_obs[strict_obs.functional_state.eq(origin)], label, destinations))
pd.DataFrame(strict_results).to_csv(OUT / "table_strict_function_transition_models.csv", index=False, encoding="utf-8-sig")

# First observed deterioration among participants without any limitation in 2013,
# using the primary (any difficulty) functional definition and time-updated exposure.
primary = pd.read_csv(SOURCE / "functional_multistate_intervals_weighted.csv", dtype={"ID": str})
baseline_no = set(primary.loc[primary.wave.eq(2) & primary.functional_state.eq(0), "ID"])
risk = primary[primary.ID.isin(baseline_no)].sort_values(["ID", "wave"]).copy()
keep = []
for _, g in risk.groupby("ID", sort=False):
    for idx, row in g.iterrows():
        if row.functional_state != 0:
            break
        keep.append(idx)
        if row.destination_state in [1, 2, 3]:
            break
incident = risk.loc[keep].copy()
incident["destination_state"] = incident.destination_state.astype(float)
incident_result = fit_model(incident, "Baseline no limitation to first observed functional deterioration", [1, 2])
incident_result["baseline_no_limitation_people"] = len(baseline_no)
pd.DataFrame([incident_result]).to_csv(OUT / "table_incident_first_decline.csv", index=False, encoding="utf-8-sig")

qa = {
    "strict_definition": "ADL limitation requires >=2 of 6 ADL difficulties; IADL-only limitation requires <2 ADL and >=2 of 5 IADL difficulties.",
    "strict_observed_intervals": int(len(strict_obs)), "strict_people": int(strict_obs.ID.nunique()),
    "strict_state_counts": strict_obs.functional_state.value_counts().sort_index().astype(int).to_dict(),
    "strict_models_converged": bool(pd.DataFrame(strict_results).converged.all()), "strict_weight_model": wqa,
    "incident_definition": "No limitation in 2013 under the primary any-difficulty definition; person-interval follow-up stops at first IADL/ADL limitation or death.",
    "incident_baseline_people": int(len(baseline_no)), "incident_intervals": int(len(incident)),
    "incident_events": int(incident.destination_state.isin([1, 2]).sum()), "incident_deaths": int(incident.destination_state.eq(3).sum()),
}
(OUT / "strict_and_incident_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))
print(pd.DataFrame(strict_results).to_string(index=False))
print(pd.DataFrame([incident_result]).to_string(index=False))

