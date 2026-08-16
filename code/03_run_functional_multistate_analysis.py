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
OUT = PROJECT / "outputs" / "comorbidity_state"
STATE_LABELS = {0: "Neither", 1: "Digestive only", 2: "Depression only", 3: "Co-occurring"}
FUNCTION_LABELS = {0: "No limitation", 1: "IADL limitation only", 2: "ADL limitation", 3: "Death"}

BASE_COVARS = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + "
    "rural_hukou + current_smoker + drank_last_year + binary_covariate_missing + "
    "baseline_bmi_imp + bmi_missing + "
    "log_hh_income_imp + income_missing + other_chronic_count"
)


def prepare(df):
    x = df.copy()
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
    x["state12"] = x["digestive"] + 2 * x["cesd10"].ge(12).astype(int)
    return x


def observation_weights(all_intervals):
    x = all_intervals.copy()
    denominator_formula = (
        "outcome_observed ~ C(state) + C(functional_state) + C(interval) + " + BASE_COVARS
    )
    numerator_formula = "outcome_observed ~ C(functional_state) + C(interval)"
    den = smf.glm(denominator_formula, x, family=sm.families.Binomial()).fit()
    num = smf.glm(numerator_formula, x, family=sm.families.Binomial()).fit()
    x["observation_probability_denominator"] = np.clip(den.predict(x), 0.02, 0.995)
    x["observation_probability_numerator"] = np.clip(num.predict(x), 0.02, 0.995)
    x["iow_raw"] = x["observation_probability_numerator"] / x["observation_probability_denominator"]
    observed = x[x["outcome_observed"].eq(1)].copy()
    lo, hi = observed["iow_raw"].quantile([0.01, 0.99])
    observed["iow"] = observed["iow_raw"].clip(lo, hi)
    return observed, {
        "denominator_converged": bool(den.converged),
        "numerator_converged": bool(num.converged),
        "weight_p01": float(lo),
        "weight_p50": float(observed["iow_raw"].median()),
        "weight_p99": float(hi),
        "truncated_n": int(((observed["iow_raw"] < lo) | (observed["iow_raw"] > hi)).sum()),
    }


def fit_transition(df, origin, destinations, model_name, exposure="state", weight=None):
    risk = df[df["functional_state"].eq(origin)].copy()
    risk["event"] = risk["destination_state"].isin(destinations).astype(int)
    formula = (
        f"event ~ C({exposure}, Treatment(reference=0)) + C(interval) + " + BASE_COVARS
    )
    kwargs = {}
    if weight:
        kwargs["freq_weights"] = risk[weight]
    model = smf.glm(formula, risk, family=sm.families.Poisson(link=sm.families.links.Log()), **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(cov_type="cluster", cov_kwds={"groups": risk.loc[model.data.row_labels, "ID"]})
    used = risk.loc[model.data.row_labels].copy()
    rows = []
    ci = result.conf_int()
    terms = {}
    for code in [1, 2, 3]:
        term = f"C({exposure}, Treatment(reference=0))[T.{float(code)}]"
        if term not in result.params.index:
            term = f"C({exposure}, Treatment(reference=0))[T.{code}]"
        if term not in result.params.index:
            continue
        terms[code] = term
        rows.append({
            "model": model_name,
            "origin": FUNCTION_LABELS[origin],
            "destination": " or ".join(FUNCTION_LABELS[d] for d in destinations),
            "comparison": STATE_LABELS[code],
            "estimate_type": "transition risk ratio",
            "estimate": float(np.exp(result.params[term])),
            "ci_low": float(np.exp(ci.loc[term, 0])),
            "ci_high": float(np.exp(ci.loc[term, 1])),
            "p_value": float(result.pvalues[term]),
            "n_intervals": int(result.nobs),
            "events": int(used["event"].sum()),
            "people": int(used["ID"].nunique()),
        })
    # Directly test whether co-occurrence adds information beyond either single condition.
    for comparator_code, comparator_label in [(2, "Depression only"), (1, "Digestive only")]:
        if 3 not in terms or comparator_code not in terms:
            continue
        contrast = result.params[terms[3]] - result.params[terms[comparator_code]]
        cov = result.cov_params()
        variance = (
            cov.loc[terms[3], terms[3]] + cov.loc[terms[comparator_code], terms[comparator_code]]
            - 2 * cov.loc[terms[3], terms[comparator_code]]
        )
        se = float(np.sqrt(max(variance, 0)))
        z = float(contrast / se) if se > 0 else np.nan
        p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        rows.append({
            "model": model_name,
            "origin": FUNCTION_LABELS[origin],
            "destination": " or ".join(FUNCTION_LABELS[d] for d in destinations),
            "comparison": f"Co-occurring vs {comparator_label}",
            "estimate_type": "ratio of transition risk ratios",
            "estimate": float(np.exp(contrast)),
            "ci_low": float(np.exp(contrast - 1.96 * se)),
            "ci_high": float(np.exp(contrast + 1.96 * se)),
            "p_value": p,
            "n_intervals": int(result.nobs),
            "events": int(used["event"].sum()),
            "people": int(used["ID"].nunique()),
        })
    return result, rows


raw_all = pd.read_csv(OUT / "functional_multistate_intervals_all.csv", dtype={"ID": str})
all_intervals = prepare(raw_all)
observed, weight_qa = observation_weights(all_intervals)
observed.to_csv(OUT / "functional_multistate_intervals_weighted.csv", index=False, encoding="utf-8-sig")

# Descriptive transition probabilities by exposure state and origin state.
descriptive = (
    observed.groupby(["functional_state", "destination_state", "state"], observed=True)
    .size().rename("n").reset_index()
)
descriptive["risk_within_origin_exposure"] = descriptive["n"] / descriptive.groupby(
    ["functional_state", "state"]
)["n"].transform("sum")
descriptive["origin_label"] = descriptive["functional_state"].map(FUNCTION_LABELS)
descriptive["destination_label"] = descriptive["destination_state"].map(FUNCTION_LABELS)
descriptive["exposure_label"] = descriptive["state"].map(STATE_LABELS)
descriptive.to_csv(OUT / "table_multistate_descriptive_risks.csv", index=False, encoding="utf-8-sig")

# Direct and clinically grouped transitions. Other destinations remain competing outcomes.
transition_specs = [
    (0, [1], "No limitation to IADL limitation"),
    (0, [2], "No limitation to ADL limitation"),
    (0, [1, 2], "No limitation to any functional deterioration"),
    (0, [3], "No limitation to death"),
    (1, [0], "IADL limitation to no limitation (recovery)"),
    (1, [2], "IADL limitation to ADL limitation"),
    (1, [3], "IADL limitation to death"),
    (2, [0], "ADL limitation to no limitation (full recovery)"),
    (2, [1], "ADL limitation to IADL only (partial recovery)"),
    (2, [0, 1], "ADL limitation to functional improvement"),
    (2, [3], "ADL limitation to death"),
]

rows = []
convergence = {}
for origin, dests, label in transition_specs:
    result, out = fit_transition(observed, origin, dests, "Primary unweighted")
    convergence[f"unweighted:{label}"] = bool(result.converged)
    rows.extend(out)
    result_w, out_w = fit_transition(observed, origin, dests, "Inverse-observation weighted", weight="iow")
    convergence[f"weighted:{label}"] = bool(result_w.converged)
    rows.extend(out_w)

    # Alternative depressive-symptom threshold.
    result_12, out_12 = fit_transition(observed, origin, dests, "CES-D threshold 12", exposure="state12")
    convergence[f"threshold12:{label}"] = bool(result_12.converged)
    rows.extend(out_12)

results = pd.DataFrame(rows)
results.to_csv(OUT / "table_multistate_transition_models.csv", index=False, encoding="utf-8-sig")

# Sensitivity excluding harmonized disease-dispute intervals. 2020 has no harmonized dispute flag.
clean = observed[observed["digestive_dispute"].eq(0)].copy()
clean_rows = []
for origin, dests, label in transition_specs:
    result, out = fit_transition(clean, origin, dests, "Exclude digestive dispute flags")
    convergence[f"clean:{label}"] = bool(result.converged)
    clean_rows.extend(out)
pd.DataFrame(clean_rows).to_csv(
    OUT / "table_multistate_dispute_sensitivity.csv", index=False, encoding="utf-8-sig"
)

qa = {
    "eligible_origin_intervals": int(len(all_intervals)),
    "observed_destination_intervals": int(len(observed)),
    "unique_people": int(observed["ID"].nunique()),
    "deaths": int(observed["destination_state"].eq(3).sum()),
    "observation_weight_model": weight_qa,
    "all_models_converged": bool(all(convergence.values())),
    "model_convergence": convergence,
    "integrity_checks": {
        "duplicate_id_origin_wave": int(observed.duplicated(["ID", "wave"]).sum()),
        "invalid_origin_states": int((~observed["functional_state"].isin([0, 1, 2])).sum()),
        "invalid_destination_states": int((~observed["destination_state"].isin([0, 1, 2, 3])).sum()),
        "missing_exposure": int(observed["state"].isna().sum()),
        "nonpositive_weights": int(observed["iow"].le(0).sum()),
    },
    "interpretation": (
        "Modified-Poisson transition risk ratios compare each destination risk across time-varying "
        "digestive-depression states. Other destinations, including death, are retained as competing outcomes."
    ),
}
(OUT / "functional_multistate_model_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))

