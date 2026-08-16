from pathlib import Path
import json
import sys
import warnings


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "outputs" / "comorbidity_state"
OUT = PROJECT / "outputs" / "digestive_depression_multistate"
OUT.mkdir(parents=True, exist_ok=True)

STATE_LABELS = {
    0: "Neither",
    1: "Digestive only",
    2: "Depression only",
    3: "Co-occurring",
}
FUNCTION_LABELS = {
    0: "No limitation",
    1: "IADL limitation only",
    2: "ADL limitation",
    3: "Death",
}
KEY_TRANSITIONS = [
    (0, [2], "No limitation to ADL limitation"),
    (1, [2], "IADL limitation to ADL limitation"),
    (1, [0], "IADL limitation to no limitation (recovery)"),
    (2, [0, 1], "ADL limitation to functional improvement"),
]
BASE_COVARS = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + "
    "rural_hukou + current_smoker + drank_last_year + binary_covariate_missing + "
    "baseline_bmi_imp + bmi_missing + "
    "log_hh_income_imp + income_missing + other_chronic_count"
)
COMPLETE_CASE_COVARS = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + "
    "current_smoker + drank_last_year + baseline_bmi_imp + log_hh_income_imp + "
    "other_chronic_count"
)


def prepare(df):
    x = df.copy()
    x["baseline_age_c"] = x["baseline_age"] - 60
    x["education_cat"] = x["education_code"].fillna(-1).astype(str)
    x["marital_cat"] = x["marital_code"].fillna(-1).astype(str)
    x["binary_covariate_missing"] = x[["rural_hukou", "current_smoker", "drank_last_year"]].isna().any(axis=1).astype(int)
    for variable in ["female", "rural_hukou", "current_smoker", "drank_last_year"]:
        x[variable] = x[variable].fillna(x[variable].mode(dropna=True).iloc[0])
    x["bmi_missing"] = x["baseline_bmi"].isna().astype(int)
    x["baseline_bmi_imp"] = x["baseline_bmi"].fillna(x["baseline_bmi"].median())
    x["income_missing"] = x["log_hh_income"].isna().astype(int)
    x["log_hh_income_imp"] = x["log_hh_income"].fillna(x["log_hh_income"].median())
    return x


def build_observation_weights(all_intervals):
    denominator = smf.glm(
        "outcome_observed ~ C(state) + C(functional_state) + C(interval) + " + BASE_COVARS,
        all_intervals,
        family=sm.families.Binomial(),
    ).fit()
    numerator = smf.glm(
        "outcome_observed ~ C(functional_state) + C(interval)",
        all_intervals,
        family=sm.families.Binomial(),
    ).fit()
    x = all_intervals.copy()
    x["p_observed_den"] = np.clip(denominator.predict(x), 0.02, 0.995)
    x["p_observed_num"] = np.clip(numerator.predict(x), 0.02, 0.995)
    x["iow_raw"] = x["p_observed_num"] / x["p_observed_den"]
    x = x[x["outcome_observed"].eq(1)].copy()
    lo, hi = x["iow_raw"].quantile([0.01, 0.99])
    x["iow"] = x["iow_raw"].clip(lo, hi)
    return x, denominator, numerator


def fit_transition(df, origin, destinations, weight="iow", covars=BASE_COVARS):
    risk = df[df["functional_state"].eq(origin)].copy()
    risk["event"] = risk["destination_state"].isin(destinations).astype(int)
    formula = "event ~ C(state, Treatment(reference=0)) + C(interval) + " + covars
    kwargs = {"freq_weights": risk[weight]} if weight else {}
    model = smf.glm(
        formula,
        risk,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": risk.loc[model.data.row_labels, "ID"]},
        )
    used = risk.loc[model.data.row_labels].copy()
    return used, result


def term_for(result, code):
    for candidate in [
        f"C(state, Treatment(reference=0))[T.{float(code)}]",
        f"C(state, Treatment(reference=0))[T.{code}]",
    ]:
        if candidate in result.params.index:
            return candidate
    raise KeyError(code)


def extract_rr(result, model_name, transition, risk):
    ci = result.conf_int()
    rows = []
    for code in [1, 2, 3]:
        term = term_for(result, code)
        rows.append({
            "analysis": model_name,
            "transition": transition,
            "comparison": STATE_LABELS[code],
            "estimate": float(np.exp(result.params[term])),
            "ci_low": float(np.exp(ci.loc[term, 0])),
            "ci_high": float(np.exp(ci.loc[term, 1])),
            "p_value": float(result.pvalues[term]),
            "n_intervals": int(result.nobs),
            "events": int(risk["event"].sum()),
            "people": int(risk["ID"].nunique()),
        })
    t3, t2 = term_for(result, 3), term_for(result, 2)
    contrast = result.params[t3] - result.params[t2]
    cov = result.cov_params()
    variance = cov.loc[t3, t3] + cov.loc[t2, t2] - 2 * cov.loc[t3, t2]
    se = float(np.sqrt(max(variance, 0)))
    rows.append({
        "analysis": model_name,
        "transition": transition,
        "comparison": "Co-occurring vs Depression only",
        "estimate": float(np.exp(contrast)),
        "ci_low": float(np.exp(contrast - 1.96 * se)),
        "ci_high": float(np.exp(contrast + 1.96 * se)),
        "p_value": float(2 * (1 - norm.cdf(abs(contrast / se)))) if se else np.nan,
        "n_intervals": int(result.nobs),
        "events": int(risk["event"].sum()),
        "people": int(risk["ID"].nunique()),
    })
    return rows


def additive_interaction(result, transition):
    t1, t2, t3 = (term_for(result, code) for code in [1, 2, 3])
    beta = result.params
    rr10, rr01, rr11 = np.exp(beta[t1]), np.exp(beta[t2]), np.exp(beta[t3])
    reri = rr11 - rr10 - rr01 + 1
    gradient = pd.Series(0.0, index=beta.index)
    gradient[t1] = -rr10
    gradient[t2] = -rr01
    gradient[t3] = rr11
    variance = float(gradient @ result.cov_params() @ gradient)
    se = np.sqrt(max(variance, 0))
    return {
        "transition": transition,
        "rr_digestive_only": float(rr10),
        "rr_depression_only": float(rr01),
        "rr_cooccurring": float(rr11),
        "reri": float(reri),
        "ci_low": float(reri - 1.96 * se),
        "ci_high": float(reri + 1.96 * se),
        "p_value": float(2 * (1 - norm.cdf(abs(reri / se)))) if se else np.nan,
    }


def standardized_risks(risk, result, transition):
    from patsy import build_design_matrices

    weights = risk["iow"].to_numpy(float)
    weights = weights / weights.sum()
    beta = result.params.to_numpy(float)
    covariance = result.cov_params().to_numpy(float)
    scenario = {}
    for code in [0, 1, 2, 3]:
        new = risk.copy()
        new["state"] = code
        design = np.asarray(
            build_design_matrices([result.model.data.design_info], new, return_type="dataframe")[0]
        )
        mu = np.exp(design @ beta)
        estimate = float(weights @ mu)
        gradient = (weights * mu) @ design
        variance = float(gradient @ covariance @ gradient)
        se = np.sqrt(max(variance, 0))
        scenario[code] = (estimate, gradient, se)

    rows = []
    for code in [0, 1, 2, 3]:
        estimate, _, se = scenario[code]
        rows.append({
            "transition": transition,
            "exposure": STATE_LABELS[code],
            "adjusted_risk": estimate,
            "risk_ci_low": max(0.0, estimate - 1.96 * se),
            "risk_ci_high": min(1.0, estimate + 1.96 * se),
            "risk_difference_vs_neither": estimate - scenario[0][0],
        })
    for code, label in [(3, "Co-occurring vs Depression only")]:
        difference = scenario[code][0] - scenario[2][0]
        gradient = scenario[code][1] - scenario[2][1]
        se = np.sqrt(max(float(gradient @ covariance @ gradient), 0))
        rows.append({
            "transition": transition,
            "exposure": label,
            "adjusted_risk": np.nan,
            "risk_ci_low": np.nan,
            "risk_ci_high": np.nan,
            "risk_difference_vs_neither": difference,
            "difference_ci_low": difference - 1.96 * se,
            "difference_ci_high": difference + 1.96 * se,
            "difference_p_value": float(2 * (1 - norm.cdf(abs(difference / se)))) if se else np.nan,
        })
    return rows


def first_origin_table(df):
    first = df.sort_values(["ID", "wave"]).drop_duplicates("ID").copy()
    rows = []
    continuous = [
        ("baseline_age", "Age, years"),
        ("baseline_bmi", "BMI, kg/m2"),
        ("other_chronic_count", "Other chronic conditions, count"),
        ("cesd10", "CES-D 10 score"),
    ]
    binary = [
        ("female", "Female"),
        ("rural_hukou", "Rural hukou"),
        ("current_smoker", "Current smoker"),
        ("drank_last_year", "Drank alcohol in past year"),
    ]
    for code in [0, 1, 2, 3]:
        group = first[first["state"].eq(code)]
        rows.append({"exposure": STATE_LABELS[code], "characteristic": "Participants", "value": str(len(group))})
        for variable, label in continuous:
            values = pd.to_numeric(group[variable], errors="coerce").dropna()
            rows.append({
                "exposure": STATE_LABELS[code],
                "characteristic": label,
                "value": f"{values.mean():.2f} ({values.std():.2f})",
            })
        for variable, label in binary:
            values = pd.to_numeric(group[variable], errors="coerce").dropna()
            rows.append({
                "exposure": STATE_LABELS[code],
                "characteristic": label,
                "value": f"{int(values.eq(1).sum())} ({100 * values.eq(1).mean():.1f}%)",
            })
        for origin in [0, 1, 2]:
            n = int(group["functional_state"].eq(origin).sum())
            rows.append({
                "exposure": STATE_LABELS[code],
                "characteristic": "Origin: " + FUNCTION_LABELS[origin],
                "value": f"{n} ({100 * n / len(group):.1f}%)" if len(group) else "0 (0.0%)",
            })
    return first, pd.DataFrame(rows)


raw_all = pd.read_csv(SOURCE / "functional_multistate_intervals_all.csv", dtype={"ID": str})
long = pd.read_csv(SOURCE / "comorbidity_state_long.csv", dtype={"ID": str})
wave_weights = long[["ID", "wave", "wave_weight"]].drop_duplicates(["ID", "wave"])
raw_all = raw_all.merge(wave_weights, on=["ID", "wave"], how="left", suffixes=("", "_from_long"))
if "wave_weight_from_long" in raw_all:
    raw_all["wave_weight"] = raw_all["wave_weight"].fillna(raw_all["wave_weight_from_long"])
    raw_all = raw_all.drop(columns="wave_weight_from_long")

complete_variables = [
    "baseline_age", "female", "education_code", "marital_code", "rural_hukou",
    "current_smoker", "drank_last_year", "baseline_bmi", "log_hh_income", "other_chronic_count",
]
raw_all["complete_baseline_covariates"] = raw_all[complete_variables].notna().all(axis=1).astype(int)

all_intervals = prepare(raw_all)
observed, observation_denominator, observation_numerator = build_observation_weights(all_intervals)

# Normalize the origin-wave survey weight within interval before combining it with IOW.
observed["survey_weight_norm"] = observed["wave_weight"] / observed.groupby("interval")["wave_weight"].transform("mean")
observed["survey_iow"] = observed["survey_weight_norm"] * observed["iow"]

first, table1 = first_origin_table(observed)
table1.to_csv(OUT / "table1_first_eligible_origin.csv", index=False, encoding="utf-8-sig")

exposure_distribution = (
    observed.groupby(["year", "state"]).size().rename("n").reset_index()
)
exposure_distribution["percent"] = 100 * exposure_distribution["n"] / exposure_distribution.groupby("year")["n"].transform("sum")
exposure_distribution["exposure"] = exposure_distribution["state"].map(STATE_LABELS)
exposure_distribution.to_csv(OUT / "table_exposure_distribution_by_origin_wave.csv", index=False, encoding="utf-8-sig")

primary_rows, interaction_rows, margin_rows = [], [], []
for origin, destinations, label in KEY_TRANSITIONS:
    risk, result = fit_transition(observed, origin, destinations, weight="iow")
    primary_rows.extend(extract_rr(result, "Primary inverse-observation weighted", label, risk))
    interaction_rows.append(additive_interaction(result, label))
    margin_rows.extend(standardized_risks(risk, result, label))

pd.DataFrame(primary_rows).to_csv(OUT / "table_key_transition_models.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(interaction_rows).to_csv(OUT / "table_additive_interaction.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(margin_rows).to_csv(OUT / "table_adjusted_absolute_risks.csv", index=False, encoding="utf-8-sig")

# Prespecified robustness analyses for the clinically central transitions.
robustness_rows = []
robustness_sets = {
    "Survey-weight and observation-weight combined": observed[observed["survey_iow"].notna() & observed["survey_iow"].gt(0)],
    "Exclude 2018-2020 interval": observed[observed["interval"].ne("2018-2020")],
    "Baseline age 60 years or older": observed[observed["baseline_age"].ge(60)],
    "Complete baseline covariates": observed[observed["complete_baseline_covariates"].eq(1)],
}
for analysis, data in robustness_sets.items():
    weight = "survey_iow" if analysis.startswith("Survey-weight") else "iow"
    for origin, destinations, label in KEY_TRANSITIONS:
        covars = COMPLETE_CASE_COVARS if analysis == "Complete baseline covariates" else BASE_COVARS
        risk, result = fit_transition(data, origin, destinations, weight=weight, covars=covars)
        robustness_rows.extend(extract_rr(result, analysis, label, risk))
pd.DataFrame(robustness_rows).to_csv(OUT / "table_key_transition_robustness.csv", index=False, encoding="utf-8-sig")

raw_first = (
    raw_all[raw_all["outcome_observed"].eq(1)]
    .sort_values(["ID", "wave"])
    .drop_duplicates("ID")
)
missingness = []
for variable in complete_variables:
    missingness.append({
        "variable": variable,
        "missing_n": int(raw_first[variable].isna().sum()),
        "missing_percent": float(100 * raw_first[variable].isna().mean()),
    })
pd.DataFrame(missingness).to_csv(OUT / "table_baseline_missingness.csv", index=False, encoding="utf-8-sig")

key = pd.DataFrame(primary_rows)
gate_checks = {
    "adequate_unique_participants_ge_5000": int(observed["ID"].nunique()) >= 5000,
    "adequate_key_transition_events_ge_200": all(
        key.groupby("transition")["events"].first().ge(200)
    ),
    "cooccurrence_predicts_no_limitation_to_adl": bool(
        ((key["transition"].eq("No limitation to ADL limitation")) &
         (key["comparison"].eq("Co-occurring")) &
         (key["ci_low"].gt(1))).any()
    ),
    "cooccurrence_adds_beyond_depression_for_iadl_to_adl": bool(
        ((key["transition"].eq("IADL limitation to ADL limitation")) &
         (key["comparison"].eq("Co-occurring vs Depression only")) &
         (key["ci_low"].gt(1))).any()
    ),
    "cooccurrence_reduces_iadl_recovery": bool(
        ((key["transition"].eq("IADL limitation to no limitation (recovery)")) &
         (key["comparison"].eq("Co-occurring")) &
         (key["ci_high"].lt(1))).any()
    ),
}
value_gate = {
    "decision": "CONTINUE" if all(gate_checks.values()) else "STOP_OR_REDESIGN",
    "checks": gate_checks,
    "observed_intervals": int(len(observed)),
    "unique_participants": int(observed["ID"].nunique()),
    "interpretation": (
        "Continue only if sample size, event support, severe deterioration, incremental information beyond "
        "depression alone, and impaired early recovery are all supported."
    ),
}
(OUT / "research_value_gate.json").write_text(json.dumps(value_gate, indent=2), encoding="utf-8")

qa = {
    "input_rows": int(len(raw_all)),
    "observed_rows": int(len(observed)),
    "unique_participants": int(observed["ID"].nunique()),
    "duplicate_person_origin_wave": int(observed.duplicated(["ID", "wave"]).sum()),
    "survey_weight_missing": int(observed["wave_weight"].isna().sum()),
    "combined_weight_nonpositive": int(observed["survey_iow"].le(0).sum()),
    "combined_weight_percentiles": {
        str(p): float(observed["survey_iow"].quantile(p)) for p in [0.01, 0.5, 0.99]
    },
    "observation_models_converged": bool(observation_denominator.converged and observation_numerator.converged),
    "value_gate": value_gate,
}
(OUT / "completion_analysis_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))

