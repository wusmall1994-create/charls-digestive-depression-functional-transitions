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
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm


SOURCE = ROOT / "outputs" / "comorbidity_state"
OUT = ROOT / "outputs" / "digestive_depression_value_upgrade_2026-09-05"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + "
    "current_smoker + drank_last_year + baseline_bmi_imp + bmi_missing + "
    "log_hh_income_imp + income_missing + other_chronic_count"
)


def prepare(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["baseline_age_c"] = x["baseline_age"] - 60
    x["education_cat"] = x["education_code"].fillna(-1).astype(str)
    x["marital_cat"] = x["marital_code"].fillna(-1).astype(str)
    for v in ["female", "rural_hukou", "current_smoker", "drank_last_year"]:
        x[v] = x[v].fillna(x[v].mode(dropna=True).iloc[0])
    x["bmi_missing"] = x["baseline_bmi"].isna().astype(int)
    x["baseline_bmi_imp"] = x["baseline_bmi"].fillna(x["baseline_bmi"].median())
    x["income_missing"] = x["log_hh_income"].isna().astype(int)
    x["log_hh_income_imp"] = x["log_hh_income"].fillna(x["log_hh_income"].median())
    return x


def observed_probability(model, matrix: pd.DataFrame, observed: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(matrix)
    class_to_col = {int(k): i for i, k in enumerate(model.classes_)}
    return np.asarray([probs[i, class_to_col[int(a)]] for i, a in enumerate(observed)])


def contrast(fit, a=3, b=2):
    def term(code):
        return next(
            t for t in [
                f"C(state, Treatment(reference=0))[T.{code}]",
                f"C(state, Treatment(reference=0))[T.{float(code)}]",
            ] if t in fit.params.index
        )
    ta, tb = term(a), term(b)
    beta = float(fit.params[ta] - fit.params[tb])
    cov = fit.cov_params()
    se = float(np.sqrt(max(cov.loc[ta, ta] + cov.loc[tb, tb] - 2 * cov.loc[ta, tb], 0)))
    return float(np.exp(beta)), float(np.exp(beta - 1.96 * se)), float(np.exp(beta + 1.96 * se)), float(2 * (1 - norm.cdf(abs(beta / se)))) if se else np.nan


all_intervals = prepare(pd.read_csv(SOURCE / "functional_multistate_intervals_all.csv", dtype={"ID": str}))
observed = pd.read_csv(SOURCE / "functional_multistate_intervals_weighted.csv", dtype={"ID": str})
long = pd.read_csv(SOURCE / "comorbidity_state_long.csv", dtype={"ID": str})

lag = long[["ID", "wave", "state", "cesd10"]].copy()
lag["wave"] = lag["wave"] + 1
lag = lag.rename(columns={"state": "lag_state", "cesd10": "lag_cesd10"})
all_intervals = all_intervals.merge(lag, on=["ID", "wave"], how="left", validate="many_to_one")

# Only records with an observed exposure history from the preceding wave enter the treatment model.
x = all_intervals[
    all_intervals["state"].notna() & all_intervals["lag_state"].notna() & all_intervals["functional_state"].notna()
].copy()
x["state"] = x["state"].astype(int)
x["lag_state"] = x["lag_state"].astype(int)
x["lag_cesd_missing"] = x["lag_cesd10"].isna().astype(int)
x["lag_cesd10_imp"] = x["lag_cesd10"].fillna(x["lag_cesd10"].median())

numerator_formula = "0 + C(interval) + C(lag_state) + " + BASE
denominator_formula = numerator_formula + " + C(functional_state) + lag_cesd10_imp + lag_cesd_missing"
xn = dmatrix(numerator_formula, x, return_type="dataframe")
xd = dmatrix(denominator_formula, x, return_type="dataframe")
common_index = xn.index.intersection(xd.index)
x = x.loc[common_index].copy()
xn = xn.loc[common_index]
xd = xd.loc[common_index]
y = x["state"].to_numpy()

num_model = LogisticRegression(max_iter=3000, solver="lbfgs", C=1e6).fit(xn, y)
den_model = LogisticRegression(max_iter=3000, solver="lbfgs", C=1e6).fit(xd, y)
x["p_num"] = np.clip(observed_probability(num_model, xn, y), 0.01, 0.99)
x["p_den"] = np.clip(observed_probability(den_model, xd, y), 0.01, 0.99)
x["sw_component"] = x["p_num"] / x["p_den"]
x = x.sort_values(["ID", "wave"])
x["sw_raw"] = x.groupby("ID", sort=False)["sw_component"].cumprod()
lo, hi = x["sw_raw"].quantile([0.01, 0.99])
x["sw"] = x["sw_raw"].clip(lo, hi)

weighted = observed.merge(x[["ID", "wave", "sw", "sw_raw", "p_num", "p_den"]], on=["ID", "wave"], how="inner")
weighted["msm_weight"] = weighted["iow"] * weighted["sw"]

definitions = [
    (0, [2], "No limitation to ADL limitation"),
    (1, [2], "IADL limitation to ADL limitation"),
    (1, [0], "IADL limitation to no limitation (recovery)"),
    (2, [0, 1], "ADL limitation to functional improvement"),
]
rows = []
for origin, destinations, label in definitions:
    risk = weighted[weighted["functional_state"].eq(origin)].copy()
    risk["event"] = risk["destination_state"].isin(destinations).astype(int)
    formula = "event ~ C(state, Treatment(reference=0)) + C(interval) + " + BASE
    model = smf.glm(
        formula,
        risk,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        freq_weights=risk["msm_weight"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": risk.loc[model.data.row_labels, "ID"]})
    rr, low, high, p = contrast(fit)
    rows.append({
        "transition": label,
        "comparison": "Co-occurring vs depressive symptoms only",
        "risk_ratio": rr,
        "ci_low": low,
        "ci_high": high,
        "p_value": p,
        "n_intervals": int(fit.nobs),
        "events": int(risk["event"].sum()),
        "people": int(risk["ID"].nunique()),
    })

results = pd.DataFrame(rows)
results.to_csv(OUT / "table_marginal_structural_sensitivity.csv", index=False, encoding="utf-8-sig")
weighted[["ID", "wave", "state", "functional_state", "destination_state", "sw_raw", "sw", "iow", "msm_weight"]].to_csv(
    OUT / "marginal_structural_weighted_intervals.csv", index=False, encoding="utf-8-sig"
)

diag = {
    "treatment_model_intervals": int(len(x)),
    "weighted_outcome_intervals": int(len(weighted)),
    "people": int(weighted["ID"].nunique()),
    "stabilized_weight_raw": {
        "mean": float(x["sw_raw"].mean()), "sd": float(x["sw_raw"].std()),
        "min": float(x["sw_raw"].min()), "p01": float(lo), "median": float(x["sw_raw"].median()),
        "p99": float(hi), "max": float(x["sw_raw"].max()),
    },
    "minimum_denominator_probability": float(x["p_den"].min()),
    "proportion_denominator_probability_below_0_05": float((x["p_den"] < 0.05).mean()),
    "estimand_boundary": (
        "Sensitivity analysis using measured prior exposure, depressive-symptom severity and functional-state history. "
        "It does not eliminate residual time-varying confounding or identify an intervention effect."
    ),
}
(OUT / "marginal_structural_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
print(json.dumps(diag, indent=2))
print(results.to_string(index=False))


