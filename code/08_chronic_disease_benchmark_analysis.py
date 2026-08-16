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
from patsy import build_design_matrices


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["CHARLS_DATA_DIR"]).expanduser().resolve()
SOURCE = ROOT / "outputs" / "comorbidity_state"
OUT = ROOT / "outputs" / "digestive_depression_multistate"
HARMONIZED = DATA / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta"

INDEXES = {
    "Digestive disease": "digestive",
    "Hypertension": "hypertension",
    "Diabetes": "diabetes",
}
STATE_LABELS = {0: "Neither", 1: "Index disease only", 2: "Depression only", 3: "Co-occurring"}
KEY_TRANSITIONS = [
    (0, [2], "No limitation to ADL limitation"),
    (1, [2], "IADL limitation to ADL limitation"),
    (1, [0], "IADL limitation to no limitation (recovery)"),
    (2, [0, 1], "ADL limitation to functional improvement"),
]
BASE_COVARS = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + "
    "rural_hukou + current_smoker + drank_last_year + binary_covariate_missing + "
    "baseline_bmi_imp + bmi_missing + log_hh_income_imp + income_missing + index_other_chronic_count"
)


def numeric_binary(series):
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[series.eq(0)] = 0.0
    out.loc[series.eq(1)] = 1.0
    return out


def term_for(result, code):
    for candidate in [
        f"C(index_state, Treatment(reference=0))[T.{float(code)}]",
        f"C(index_state, Treatment(reference=0))[T.{code}]",
    ]:
        if candidate in result.params.index:
            return candidate
    raise KeyError(code)


def fit_full_state(data, origin, destinations):
    risk = data[data["functional_state"].eq(origin)].copy()
    risk["event"] = risk["destination_state"].isin(destinations).astype(int)
    formula = "event ~ C(index_state, Treatment(reference=0)) + C(interval) + " + BASE_COVARS
    model = smf.glm(
        formula, risk, family=sm.families.Poisson(link=sm.families.links.Log()),
        freq_weights=risk["iow"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(cov_type="cluster", cov_kwds={"groups": risk.loc[model.data.row_labels, "ID"]})
    used = risk.loc[model.data.row_labels].copy()
    return used, result


def direct_contrast(result):
    t3, t2 = term_for(result, 3), term_for(result, 2)
    log_rr = float(result.params[t3] - result.params[t2])
    cov = result.cov_params()
    var = float(cov.loc[t3, t3] + cov.loc[t2, t2] - 2 * cov.loc[t3, t2])
    se = np.sqrt(max(var, 0))
    return log_rr, se


def standardized_risk_difference(used, result):
    weights = used["iow"].to_numpy(float).copy()
    weights /= weights.sum()
    beta = result.params.to_numpy(float)
    covariance = result.cov_params().to_numpy(float)
    risks, gradients = {}, {}
    for code in [2, 3]:
        new = used.copy()
        new["index_state"] = code
        design = np.asarray(build_design_matrices([result.model.data.design_info], new)[0])
        mu = np.exp(design @ beta)
        risks[code] = float(weights @ mu)
        gradients[code] = (weights * mu) @ design
    grad = gradients[3] - gradients[2]
    rd = risks[3] - risks[2]
    se = float(np.sqrt(max(grad @ covariance @ grad, 0)))
    return risks[2], risks[3], rd, se


# Pull harmonized hypertension and diabetes statuses at the three interval origins.
hcols = ["ID"] + [f"r{w}{stem}" for w in [2, 3, 4] for stem in ["hibpe", "diabe"]]
h = pd.read_stata(HARMONIZED, columns=hcols, convert_categoricals=False)
status = []
for wave in [2, 3, 4]:
    status.append(pd.DataFrame({
        "ID": h["ID"].astype(str), "wave": wave,
        "hypertension": numeric_binary(h[f"r{wave}hibpe"]),
        "diabetes": numeric_binary(h[f"r{wave}diabe"]),
    }))
status = pd.concat(status, ignore_index=True)

base = pd.read_csv(SOURCE / "functional_multistate_intervals_weighted.csv", dtype={"ID": str})
base = base.merge(status, on=["ID", "wave"], how="left", validate="many_to_one")

rows = []
stacked_parts = []
for index_name, variable in INDEXES.items():
    data = base[base[variable].notna() & base["depression"].notna()].copy()
    data["index_disease"] = data[variable].astype(int)
    data["index_state"] = data["index_disease"] + 2 * data["depression"].astype(int)
    # The original count excludes digestive disease but includes hypertension and diabetes.
    data["index_other_chronic_count"] = data["other_chronic_count"]
    if variable in ["hypertension", "diabetes"]:
        data["index_other_chronic_count"] = (data["index_other_chronic_count"] - data[variable]).clip(lower=0)
    data["index_name"] = index_name
    stacked_parts.append(data[data["depression"].eq(1)].copy())

    for origin, destinations, label in KEY_TRANSITIONS:
        used, result = fit_full_state(data, origin, destinations)
        log_rr, se = direct_contrast(result)
        dep_risk, co_risk, rd, rd_se = standardized_risk_difference(used, result)
        rows.append({
            "index_condition": index_name,
            "transition": label,
            "ratio_cooccurring_vs_depression_only": float(np.exp(log_rr)),
            "ratio_ci_low": float(np.exp(log_rr - 1.96 * se)),
            "ratio_ci_high": float(np.exp(log_rr + 1.96 * se)),
            "ratio_p_value": float(2 * (1 - norm.cdf(abs(log_rr / se)))) if se else np.nan,
            "standardized_risk_depression_only": dep_risk,
            "standardized_risk_cooccurring": co_risk,
            "standardized_risk_difference": rd,
            "risk_difference_ci_low": rd - 1.96 * rd_se,
            "risk_difference_ci_high": rd + 1.96 * rd_se,
            "n_intervals": int(len(used)),
            "events": int(used["event"].sum()),
            "people": int(used["ID"].nunique()),
        })

results = pd.DataFrame(rows)
results.to_csv(OUT / "table_chronic_disease_benchmark.csv", index=False, encoding="utf-8-sig")

# One stacked, participant-clustered model directly compares the incremental index-disease
# association within depressed participants across the three disease families.
stacked = pd.concat(stacked_parts, ignore_index=True)
interaction_rows = []
for origin, destinations, label in KEY_TRANSITIONS:
    risk = stacked[stacked["functional_state"].eq(origin)].copy()
    risk["event"] = risk["destination_state"].isin(destinations).astype(int)
    formula = (
        "event ~ C(index_name, Treatment(reference='Digestive disease')) * index_disease + "
        "C(interval) + " + BASE_COVARS
    )
    model = smf.glm(formula, risk, family=sm.families.Poisson(link=sm.families.links.Log()), freq_weights=risk["iow"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": risk.loc[model.data.row_labels, "ID"]})
    for comparator in ["Hypertension", "Diabetes"]:
        term = f"C(index_name, Treatment(reference='Digestive disease'))[T.{comparator}]:index_disease"
        beta, se = float(fit.params[term]), float(fit.bse[term])
        # exp(-beta) is digestive incremental RR divided by comparator incremental RR.
        interaction_rows.append({
            "transition": label,
            "comparison": f"Digestive disease vs {comparator}",
            "ratio_of_incremental_risk_ratios": float(np.exp(-beta)),
            "ci_low": float(np.exp(-beta - 1.96 * se)),
            "ci_high": float(np.exp(-beta + 1.96 * se)),
            "p_heterogeneity": float(2 * (1 - norm.cdf(abs(beta / se)))) if se else np.nan,
            "n_stacked_rows": int(fit.nobs),
            "unique_people": int(risk.loc[model.data.row_labels, "ID"].nunique()),
        })

interactions = pd.DataFrame(interaction_rows)
interactions.to_csv(OUT / "table_chronic_disease_benchmark_heterogeneity.csv", index=False, encoding="utf-8-sig")

qa = {
    "input_intervals": int(len(base)),
    "hypertension_missing": int(base["hypertension"].isna().sum()),
    "diabetes_missing": int(base["diabetes"].isna().sum()),
    "benchmark_models": int(len(results)),
    "heterogeneity_contrasts": int(len(interactions)),
    "all_estimates_finite": bool(np.isfinite(results["ratio_cooccurring_vs_depression_only"]).all()),
    "all_risks_valid": bool(results[["standardized_risk_depression_only", "standardized_risk_cooccurring"]].stack().between(0, 1).all()),
    "interpretation": "Condition-specific four-state models are descriptive benchmarks. Stacked interaction models compare incremental disease associations within depressed participants and account for repeated index-condition records by clustering on participant.",
}
(OUT / "chronic_disease_benchmark_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))
print(results.to_string(index=False))
print(interactions.to_string(index=False))

