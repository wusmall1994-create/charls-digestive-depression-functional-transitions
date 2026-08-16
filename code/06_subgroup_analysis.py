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
SOURCE = PROJECT / "outputs" / "comorbidity_state" / "functional_multistate_intervals_weighted.csv"
OUT = PROJECT / "outputs" / "digestive_depression_multistate"
OUT.mkdir(parents=True, exist_ok=True)

STATE_LABELS = {0: "Neither", 1: "Digestive only", 2: "Depression only", 3: "Co-occurring"}
TRANSITIONS = [
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
MODIFIERS = {
    "female": {0: "Male", 1: "Female"},
    "age60": {0: "Baseline age <60 years", 1: "Baseline age >=60 years"},
    "rural_hukou": {0: "Urban hukou", 1: "Rural hukou"},
}


def state_term(index, code):
    for candidate in [
        f"C(state, Treatment(reference=0))[T.{float(code)}]",
        f"C(state, Treatment(reference=0))[T.{code}]",
    ]:
        if candidate in index:
            return candidate
    raise KeyError(code)


def interaction_term(index, code, modifier):
    base = state_term(index, code)
    for candidate in [f"{base}:{modifier}", f"{modifier}:{base}"]:
        if candidate in index:
            return candidate
    raise KeyError((code, modifier))


def contrast(result, coefficients):
    vector = pd.Series(0.0, index=result.params.index)
    for term, value in coefficients.items():
        vector[term] = value
    estimate = float(vector @ result.params)
    variance = float(vector @ result.cov_params() @ vector)
    se = np.sqrt(max(variance, 0))
    return estimate, se, float(2 * (1 - norm.cdf(abs(estimate / se)))) if se else np.nan


df = pd.read_csv(SOURCE, dtype={"ID": str})
df["age60"] = df["baseline_age"].ge(60).astype(int)

rows = []
convergence = {}
for modifier, labels in MODIFIERS.items():
    for origin, destinations, transition in TRANSITIONS:
        risk = df[df["functional_state"].eq(origin)].copy()
        risk["event"] = risk["destination_state"].isin(destinations).astype(int)
        formula = (
            f"event ~ C(state, Treatment(reference=0)) * {modifier} + C(interval) + " + BASE_COVARS
        )
        model = smf.glm(
            formula,
            risk,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            freq_weights=risk["iow"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": risk.loc[model.data.row_labels, "ID"]},
            )
        convergence[f"{modifier}:{transition}"] = bool(result.converged)
        index = result.params.index
        used = risk.loc[model.data.row_labels].copy()
        for level in [0, 1]:
            stratum = used[used[modifier].eq(level)]
            for code in [1, 2, 3]:
                coefs = {state_term(index, code): 1.0}
                if level == 1:
                    coefs[interaction_term(index, code, modifier)] = 1.0
                estimate, se, p = contrast(result, coefs)
                rows.append({
                    "modifier": modifier,
                    "stratum": labels[level],
                    "transition": transition,
                    "comparison": STATE_LABELS[code],
                    "estimate": float(np.exp(estimate)),
                    "ci_low": float(np.exp(estimate - 1.96 * se)),
                    "ci_high": float(np.exp(estimate + 1.96 * se)),
                    "p_value": p,
                    "n_intervals": int(len(stratum)),
                    "events": int(stratum["event"].sum()),
                })
            direct = {state_term(index, 3): 1.0, state_term(index, 2): -1.0}
            if level == 1:
                direct[interaction_term(index, 3, modifier)] = 1.0
                direct[interaction_term(index, 2, modifier)] = -1.0
            estimate, se, p = contrast(result, direct)
            interaction = {
                interaction_term(index, 3, modifier): 1.0,
                interaction_term(index, 2, modifier): -1.0,
            }
            _, _, p_interaction = contrast(result, interaction)
            rows.append({
                "modifier": modifier,
                "stratum": labels[level],
                "transition": transition,
                "comparison": "Co-occurring vs Depression only",
                "estimate": float(np.exp(estimate)),
                "ci_low": float(np.exp(estimate - 1.96 * se)),
                "ci_high": float(np.exp(estimate + 1.96 * se)),
                "p_value": p,
                "p_interaction": p_interaction,
                "n_intervals": int(len(stratum)),
                "events": int(stratum["event"].sum()),
            })

results = pd.DataFrame(rows)
results.to_csv(OUT / "table_subgroup_transition_models.csv", index=False, encoding="utf-8-sig")
qa = {
    "models": len(convergence),
    "all_models_converged": bool(all(convergence.values())),
    "duplicate_output_rows": int(results.duplicated(["modifier", "stratum", "transition", "comparison"]).sum()),
    "invalid_confidence_intervals": int((results["ci_low"] > results["ci_high"]).sum()),
    "convergence": convergence,
}
(OUT / "subgroup_analysis_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))

