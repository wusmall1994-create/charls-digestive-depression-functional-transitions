from pathlib import Path
import os
import json
import sys
import warnings


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["CHARLS_DATA_DIR"]).expanduser().resolve()
SOURCE = ROOT / "outputs" / "comorbidity_state"
OUT = ROOT / "outputs" / "digestive_depression_multistate"
RAW_SLEEP = {
    2: DATA / "2013" / "CHARLS2013_Dataset" / "Health_Status_and_Functioning.dta",
    3: DATA / "2015" / "CHARLS2015r" / "Health_Status_and_Functioning.dta",
}

BASE_COVARS = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + "
    "current_smoker + drank_last_year + binary_covariate_missing + baseline_bmi_imp + "
    "bmi_missing + log_hh_income_imp + income_missing + other_chronic_count"
)


def contrast(result, exposure="state2013"):
    candidates = {
        code: [
            f"C({exposure}, Treatment(reference=0))[T.{float(code)}]",
            f"C({exposure}, Treatment(reference=0))[T.{code}]",
        ] for code in [2, 3]
    }
    terms = {}
    for code, options in candidates.items():
        terms[code] = next(x for x in options if x in result.params.index)
    log_rr = float(result.params[terms[3]] - result.params[terms[2]])
    cov = result.cov_params()
    var = float(cov.loc[terms[3], terms[3]] + cov.loc[terms[2], terms[2]] - 2 * cov.loc[terms[3], terms[2]])
    se = np.sqrt(max(var, 0))
    return float(np.exp(log_rr)), float(np.exp(log_rr - 1.96 * se)), float(np.exp(log_rr + 1.96 * se)), log_rr


# Sleep at 2013 precedes the 2013 exposure-to-2015 mediator association adjustment;
# sleep at 2015 precedes functional transitions observed from 2015 to 2018.
sleep_frames = []
for wave, path in RAW_SLEEP.items():
    frame = pd.read_stata(path, columns=["ID", "da049"], convert_categoricals=False)
    frame["ID"] = frame["ID"].astype(str)
    frame["wave"] = wave
    frame["sleep_hours"] = pd.to_numeric(frame["da049"], errors="coerce")
    frame.loc[~frame["sleep_hours"].between(0, 24), "sleep_hours"] = np.nan
    sleep_frames.append(frame[["ID", "wave", "sleep_hours"]])
sleep = pd.concat(sleep_frames, ignore_index=True)
sleep13 = sleep[sleep["wave"].eq(2)][["ID", "sleep_hours"]].rename(columns={"sleep_hours": "sleep2013"})
sleep15 = sleep[sleep["wave"].eq(3)][["ID", "sleep_hours"]].rename(columns={"sleep_hours": "sleep2015"})

long = pd.read_csv(SOURCE / "functional_multistate_long.csv", dtype={"ID": str})
state13 = long[long["wave"].eq(2)][["ID", "state", "functional_state"]].rename(
    columns={"state": "state2013", "functional_state": "function2013"}
)
intervals = pd.read_csv(SOURCE / "functional_multistate_intervals_weighted.csv", dtype={"ID": str})
data = intervals[intervals["wave"].eq(3)].copy()
data = data.merge(state13, on="ID", how="inner", validate="many_to_one")
data = data.merge(sleep13, on="ID", how="left", validate="many_to_one")
data = data.merge(sleep15, on="ID", how="left", validate="many_to_one")
data["sleep13_missing"] = data["sleep2013"].isna().astype(int)
data["sleep2013_imp"] = data["sleep2013"].fillna(data["sleep2013"].median())
data["sleep15_category"] = np.select(
    [data["sleep2015"].lt(6), data["sleep2015"].between(6, 8, inclusive="both"), data["sleep2015"].gt(8)],
    [0, 1, 2], default=3,
).astype(int)

# Exposure-to-mediator link: 2013 co-occurrence vs depression only and short sleep in 2015,
# controlling 2013 sleep and 2013 functional state.
med = data[data["state2013"].isin([2, 3])].drop_duplicates("ID").copy()
med["cooccurring2013"] = med["state2013"].eq(3).astype(int)
med["short_sleep2015"] = med["sleep2015"].lt(6).astype(float).where(med["sleep2015"].notna())
med_formula = (
    "short_sleep2015 ~ cooccurring2013 + sleep2013_imp + sleep13_missing + C(function2013) + " + BASE_COVARS
)
med_glm = smf.glm(med_formula, med, family=sm.families.Binomial())
med_model = med_glm.fit(
    cov_type="cluster", cov_kwds={"groups": med.loc[med_glm.data.row_labels, "ID"]}
)
med_beta, med_se = float(med_model.params["cooccurring2013"]), float(med_model.bse["cooccurring2013"])

# Outcome link at the IADL stage: 2013 exposure, sleep measured in 2015, and the
# subsequent 2015-2018 transition. This is a temporally ordered attenuation analysis,
# not a causal indirect-effect estimator.
iadl = data[data["functional_state"].eq(1) & data["state2013"].notna()].copy()
rows = []
for label, destinations in [
    ("IADL limitation to ADL limitation", [2]),
    ("IADL limitation to no limitation (recovery)", [0]),
]:
    iadl["event"] = iadl["destination_state"].isin(destinations).astype(int)
    base_formula = "event ~ C(state2013, Treatment(reference=0)) + " + BASE_COVARS
    sleep_formula = base_formula + " + C(sleep15_category, Treatment(reference=1))"
    fits = []
    for formula in [base_formula, sleep_formula]:
        model = smf.glm(formula, iadl, family=sm.families.Poisson(link=sm.families.links.Log()), freq_weights=iadl["iow"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.append(model.fit(cov_type="cluster", cov_kwds={"groups": iadl.loc[model.data.row_labels, "ID"]}))
    primary = contrast(fits[0])
    adjusted = contrast(fits[1])
    attenuation = np.nan
    if abs(primary[3]) > 1e-12:
        attenuation = 100 * (abs(primary[3]) - abs(adjusted[3])) / abs(primary[3])
    rows.append({
        "transition": label,
        "primary_ratio": primary[0], "primary_ci_low": primary[1], "primary_ci_high": primary[2],
        "sleep_adjusted_ratio": adjusted[0], "sleep_adjusted_ci_low": adjusted[1], "sleep_adjusted_ci_high": adjusted[2],
        "absolute_log_ratio_attenuation_percent": attenuation,
        "n_intervals": int(fits[1].nobs), "events": int(iadl["event"].sum()),
        "sleep2015_missing": int(iadl["sleep2015"].isna().sum()),
    })

outcomes = pd.DataFrame(rows)
outcomes.to_csv(OUT / "table_temporally_ordered_sleep_pathway.csv", index=False, encoding="utf-8-sig")
mediator = pd.DataFrame([{
    "comparison": "2013 co-occurring vs depression only",
    "outcome": "Short nighttime sleep (<6 h) in 2015",
    "odds_ratio": float(np.exp(med_beta)),
    "ci_low": float(np.exp(med_beta - 1.96 * med_se)),
    "ci_high": float(np.exp(med_beta + 1.96 * med_se)),
    "p_value": float(2 * (1 - norm.cdf(abs(med_beta / med_se)))),
    "n_people": int(med_model.nobs),
}])
mediator.to_csv(OUT / "table_sleep_exposure_mediator_link.csv", index=False, encoding="utf-8-sig")

qa = {
    "eligible_2015_origin_intervals": int(len(data)),
    "iadl_origin_intervals": int(len(iadl)),
    "mediator_model_n": int(med_model.nobs),
    "outcome_models": int(len(outcomes)),
    "all_estimates_finite": bool(np.isfinite(outcomes[["primary_ratio", "sleep_adjusted_ratio"]]).all().all()),
    "temporal_order": "2013 digestive-depression state -> 2015 nighttime sleep -> 2015-2018 functional transition; 2013 nighttime sleep and 2013 function included in the mediator model.",
    "estimand_boundary": "Exploratory temporally ordered attenuation; no natural indirect effect is estimated.",
}
(OUT / "temporally_ordered_sleep_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))
print(mediator.to_string(index=False))
print(outcomes.to_string(index=False))

