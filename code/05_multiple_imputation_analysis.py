from pathlib import Path
import json
import sys
import warnings


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "outputs" / "comorbidity_state"
OUT = PROJECT / "outputs" / "digestive_depression_multistate"
OUT.mkdir(parents=True, exist_ok=True)

STATE_LABELS = {0: "Neither", 1: "Digestive only", 2: "Depression only", 3: "Co-occurring"}
KEY_TRANSITIONS = [
    (0, [2], "No limitation to ADL limitation"),
    (1, [2], "IADL limitation to ADL limitation"),
    (1, [0], "IADL limitation to no limitation (recovery)"),
    (2, [0, 1], "ADL limitation to functional improvement"),
]
IMPUTE_COLUMNS = [
    "baseline_age", "female", "education_code", "marital_code", "rural_hukou",
    "current_smoker", "drank_last_year", "baseline_bmi", "log_hh_income",
    "other_chronic_count", "functional_state", "digestive", "depression", "cesd10",
]
MODEL_COVARS = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + "
    "current_smoker + drank_last_year + baseline_bmi + log_hh_income + other_chronic_count"
)
NO_INCOME_COVARS = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + "
    "current_smoker + drank_last_year + baseline_bmi_imp + bmi_missing + other_chronic_count"
)


def prepare_categories(df):
    x = df.copy()
    x["baseline_age_c"] = x["baseline_age"] - 60
    x["education_cat"] = x["education_code"].round().astype(int).astype(str)
    x["marital_cat"] = x["marital_code"].round().astype(int).astype(str)
    return x


def observation_weights(df, covars):
    den = smf.glm(
        "outcome_observed ~ C(state) + C(functional_state) + C(interval) + " + covars,
        df,
        family=sm.families.Binomial(),
    ).fit()
    num = smf.glm(
        "outcome_observed ~ C(functional_state) + C(interval)",
        df,
        family=sm.families.Binomial(),
    ).fit()
    x = df[df["outcome_observed"].eq(1)].copy()
    pden = np.clip(den.predict(x), 0.02, 0.995)
    pnum = np.clip(num.predict(x), 0.02, 0.995)
    raw = pnum / pden
    lo, hi = raw.quantile([0.01, 0.99])
    x["iow_mi"] = raw.clip(lo, hi)
    return x


def fit(df, origin, destinations, covars, weight):
    risk = df[df["functional_state"].eq(origin)].copy()
    risk["event"] = risk["destination_state"].isin(destinations).astype(int)
    formula = "event ~ C(state, Treatment(reference=0)) + C(interval) + " + covars
    model = smf.glm(
        formula,
        risk,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        freq_weights=risk[weight],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": risk.loc[model.data.row_labels, "ID"]},
        )
    return result


def term_name(index, code):
    for candidate in [
        f"C(state, Treatment(reference=0))[T.{float(code)}]",
        f"C(state, Treatment(reference=0))[T.{code}]",
    ]:
        if candidate in index:
            return candidate
    raise KeyError(code)


def pool_fits(fits, transition):
    names = list(fits[0].params.index)
    estimates = np.vstack([fit.params.reindex(names).to_numpy(float) for fit in fits])
    within = np.stack([fit.cov_params().reindex(index=names, columns=names).to_numpy(float) for fit in fits])
    m = len(fits)
    qbar = estimates.mean(axis=0)
    ubar = within.mean(axis=0)
    between = np.cov(estimates, rowvar=False, ddof=1)
    total = ubar + (1 + 1 / m) * between
    index = pd.Index(names)
    rows = []
    for code in [1, 2, 3]:
        term = term_name(index, code)
        j = index.get_loc(term)
        estimate, se = qbar[j], np.sqrt(max(total[j, j], 0))
        rows.append({
            "analysis": f"Multiple imputation (m={m})",
            "transition": transition,
            "comparison": STATE_LABELS[code],
            "estimate": float(np.exp(estimate)),
            "ci_low": float(np.exp(estimate - 1.96 * se)),
            "ci_high": float(np.exp(estimate + 1.96 * se)),
            "p_value": float(2 * (1 - norm.cdf(abs(estimate / se)))) if se else np.nan,
        })
    t3, t2 = term_name(index, 3), term_name(index, 2)
    j3, j2 = index.get_loc(t3), index.get_loc(t2)
    contrast = qbar[j3] - qbar[j2]
    variance = total[j3, j3] + total[j2, j2] - 2 * total[j3, j2]
    se = np.sqrt(max(variance, 0))
    rows.append({
        "analysis": f"Multiple imputation (m={m})",
        "transition": transition,
        "comparison": "Co-occurring vs Depression only",
        "estimate": float(np.exp(contrast)),
        "ci_low": float(np.exp(contrast - 1.96 * se)),
        "ci_high": float(np.exp(contrast + 1.96 * se)),
        "p_value": float(2 * (1 - norm.cdf(abs(contrast / se)))) if se else np.nan,
    })
    return rows


raw = pd.read_csv(SOURCE / "functional_multistate_intervals_all.csv", dtype={"ID": str})
raw["education_code"] = pd.to_numeric(raw["education_code"], errors="coerce")
raw["marital_code"] = pd.to_numeric(raw["marital_code"], errors="coerce")

# One imputation record per participant. First observed exposure/function and CES-D are
# auxiliary predictors; future destination states are deliberately not used.
person = raw.sort_values(["ID", "wave"]).drop_duplicates("ID").set_index("ID")
matrix = person[IMPUTE_COLUMNS].apply(pd.to_numeric, errors="coerce")

m = 20
fits_by_transition = {label: [] for _, _, label in KEY_TRANSITIONS}
imputation_audit = []
for iteration in range(m):
    imputer = IterativeImputer(
        max_iter=20,
        sample_posterior=True,
        random_state=20260805 + iteration,
        min_value=-np.inf,
        max_value=np.inf,
    )
    imputed_values = imputer.fit_transform(matrix)
    imputed = pd.DataFrame(imputed_values, index=matrix.index, columns=matrix.columns)
    # Only continuous covariates are taken from the imputer. The few missing discrete
    # covariates are replaced by their observed modal value to avoid invalid categories.
    covariates = person[[
        "baseline_age", "female", "education_code", "marital_code", "rural_hukou",
        "current_smoker", "drank_last_year", "baseline_bmi", "log_hh_income", "other_chronic_count",
    ]].copy()
    for variable in ["baseline_age", "baseline_bmi", "log_hh_income"]:
        covariates[variable] = covariates[variable].fillna(imputed[variable])
    for variable in ["female", "education_code", "marital_code", "rural_hukou", "current_smoker", "drank_last_year"]:
        covariates[variable] = covariates[variable].fillna(covariates[variable].mode().iloc[0])
    covariates = covariates.reset_index()
    analysis = raw.drop(columns=[c for c in covariates.columns if c != "ID"], errors="ignore").merge(
        covariates, on="ID", how="left"
    )
    analysis = prepare_categories(analysis)
    weighted = observation_weights(analysis, MODEL_COVARS)
    for origin, destinations, label in KEY_TRANSITIONS:
        fits_by_transition[label].append(fit(weighted, origin, destinations, MODEL_COVARS, "iow_mi"))
    imputation_audit.append({
        "imputation": iteration + 1,
        "mean_bmi": float(covariates["baseline_bmi"].mean()),
        "mean_log_income": float(covariates["log_hh_income"].mean()),
    })

pooled_rows = []
for _, _, label in KEY_TRANSITIONS:
    pooled_rows.extend(pool_fits(fits_by_transition[label], label))
pd.DataFrame(pooled_rows).to_csv(OUT / "table_multiple_imputation_models.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(imputation_audit).to_csv(OUT / "multiple_imputation_audit.csv", index=False, encoding="utf-8-sig")

# Sensitivity omitting household income avoids relying on either imputation or the
# missing-indicator assumption for this highly incomplete covariate.
no_income = raw.copy()
no_income["baseline_age_c"] = no_income["baseline_age"] - 60
no_income["education_cat"] = no_income["education_code"].fillna(-1).astype(str)
no_income["marital_cat"] = no_income["marital_code"].fillna(-1).astype(str)
for variable in ["female", "rural_hukou", "current_smoker", "drank_last_year"]:
    no_income[variable] = no_income[variable].fillna(no_income[variable].mode().iloc[0])
no_income["bmi_missing"] = no_income["baseline_bmi"].isna().astype(int)
no_income["baseline_bmi_imp"] = no_income["baseline_bmi"].fillna(no_income["baseline_bmi"].median())
weighted_no_income = observation_weights(no_income, NO_INCOME_COVARS)

no_income_rows = []
for origin, destinations, label in KEY_TRANSITIONS:
    result = fit(weighted_no_income, origin, destinations, NO_INCOME_COVARS, "iow_mi")
    index = result.params.index
    covariance = result.cov_params()
    for code in [1, 2, 3]:
        term = term_name(index, code)
        estimate = result.params[term]
        se = np.sqrt(covariance.loc[term, term])
        no_income_rows.append({
            "analysis": "Omit household income",
            "transition": label,
            "comparison": STATE_LABELS[code],
            "estimate": float(np.exp(estimate)),
            "ci_low": float(np.exp(estimate - 1.96 * se)),
            "ci_high": float(np.exp(estimate + 1.96 * se)),
            "p_value": float(result.pvalues[term]),
        })
    t3, t2 = term_name(index, 3), term_name(index, 2)
    contrast = result.params[t3] - result.params[t2]
    variance = covariance.loc[t3, t3] + covariance.loc[t2, t2] - 2 * covariance.loc[t3, t2]
    se = np.sqrt(max(variance, 0))
    no_income_rows.append({
        "analysis": "Omit household income",
        "transition": label,
        "comparison": "Co-occurring vs Depression only",
        "estimate": float(np.exp(contrast)),
        "ci_low": float(np.exp(contrast - 1.96 * se)),
        "ci_high": float(np.exp(contrast + 1.96 * se)),
        "p_value": float(2 * (1 - norm.cdf(abs(contrast / se)))) if se else np.nan,
    })
pd.DataFrame(no_income_rows).to_csv(OUT / "table_no_income_sensitivity.csv", index=False, encoding="utf-8-sig")

qa = {
    "imputations": m,
    "participants_imputed": int(len(person)),
    "missing_before": {variable: int(matrix[variable].isna().sum()) for variable in matrix.columns},
    "pooled_models": int(len(KEY_TRANSITIONS)),
    "all_pooled_estimates_finite": bool(np.isfinite(pd.DataFrame(pooled_rows)[["estimate", "ci_low", "ci_high"]]).all().all()),
}
(OUT / "multiple_imputation_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))

