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

RAW_FILES = {
    2: DATA / "2013" / "CHARLS2013_Dataset" / "Health_Status_and_Functioning.dta",
    3: DATA / "2015" / "CHARLS2015r" / "Health_Status_and_Functioning.dta",
    4: DATA / "2018" / "CHARLS2018r" / "Health_Status_and_Functioning.dta",
}

STATE_LABELS = {
    0: "Neither",
    1: "Digestive only",
    2: "Depression only",
    3: "Co-occurring",
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
    "baseline_bmi_imp + bmi_missing + log_hh_income_imp + income_missing + "
    "other_chronic_count"
)


def term_for(result, code):
    for candidate in [
        f"C(state, Treatment(reference=0))[T.{float(code)}]",
        f"C(state, Treatment(reference=0))[T.{code}]",
    ]:
        if candidate in result.params.index:
            return candidate
    raise KeyError(code)


def fit_transition(data, origin, destinations, add_sleep=False):
    risk = data[data["functional_state"].eq(origin)].copy()
    risk["event"] = risk["destination_state"].isin(destinations).astype(int)
    formula = "event ~ C(state, Treatment(reference=0)) + C(interval) + " + BASE_COVARS
    if add_sleep:
        formula += " + C(sleep_category, Treatment(reference=1))"
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
    used = risk.loc[model.data.row_labels].copy()
    return used, result


def direct_contrast(result):
    t3 = term_for(result, 3)
    t2 = term_for(result, 2)
    contrast = result.params[t3] - result.params[t2]
    covariance = result.cov_params()
    variance = covariance.loc[t3, t3] + covariance.loc[t2, t2] - 2 * covariance.loc[t3, t2]
    se = float(np.sqrt(max(variance, 0)))
    return {
        "estimate": float(np.exp(contrast)),
        "ci_low": float(np.exp(contrast - 1.96 * se)),
        "ci_high": float(np.exp(contrast + 1.96 * se)),
        "p_value": float(2 * (1 - norm.cdf(abs(contrast / se)))) if se else np.nan,
    }


# Add origin-wave nighttime sleep duration. Values outside 0-24 hours are treated as missing.
sleep_frames = []
for wave, path in RAW_FILES.items():
    frame = pd.read_stata(path, columns=["ID", "da049"], convert_categoricals=False)
    frame = frame.rename(columns={"da049": "night_sleep_hours"})
    frame["wave"] = wave
    sleep_frames.append(frame)
sleep = pd.concat(sleep_frames, ignore_index=True)
sleep["night_sleep_hours"] = pd.to_numeric(sleep["night_sleep_hours"], errors="coerce")
sleep.loc[~sleep["night_sleep_hours"].between(0, 24), "night_sleep_hours"] = np.nan

data = pd.read_csv(
    SOURCE / "functional_multistate_intervals_weighted.csv",
    dtype={"ID": str},
)
data = data.merge(sleep, on=["ID", "wave"], how="left", validate="many_to_one")
data["sleep_category"] = np.select(
    [
        data["night_sleep_hours"].lt(6),
        data["night_sleep_hours"].between(6, 8, inclusive="both"),
        data["night_sleep_hours"].gt(8),
    ],
    [0, 1, 2],
    default=3,
).astype(int)

sleep_rows = []
for origin, destinations, label in KEY_TRANSITIONS:
    risk_primary, primary = fit_transition(data, origin, destinations, add_sleep=False)
    risk_sleep, adjusted = fit_transition(data, origin, destinations, add_sleep=True)
    primary_contrast = direct_contrast(primary)
    sleep_contrast = direct_contrast(adjusted)
    log_primary = np.log(primary_contrast["estimate"])
    log_adjusted = np.log(sleep_contrast["estimate"])
    attenuation = np.nan
    if abs(log_primary) > 1e-12:
        attenuation = 100 * (abs(log_primary) - abs(log_adjusted)) / abs(log_primary)
    sleep_rows.append({
        "transition": label,
        "primary_estimate": primary_contrast["estimate"],
        "primary_ci_low": primary_contrast["ci_low"],
        "primary_ci_high": primary_contrast["ci_high"],
        "sleep_adjusted_estimate": sleep_contrast["estimate"],
        "sleep_adjusted_ci_low": sleep_contrast["ci_low"],
        "sleep_adjusted_ci_high": sleep_contrast["ci_high"],
        "sleep_adjusted_p_value": sleep_contrast["p_value"],
        "absolute_log_contrast_attenuation_percent": attenuation,
        "n_intervals": int(len(risk_sleep)),
        "events": int(risk_sleep["event"].sum()),
        "sleep_missing_n": int(risk_sleep["night_sleep_hours"].isna().sum()),
        "sleep_missing_percent": float(100 * risk_sleep["night_sleep_hours"].isna().mean()),
    })

sleep_table = pd.DataFrame(sleep_rows)
sleep_table.to_csv(OUT / "table_sleep_adjustment.csv", index=False, encoding="utf-8-sig")

# Complete crude event rates and model-standardized risks for the four key transitions.
margins = pd.read_csv(OUT / "table_adjusted_absolute_risks.csv")
margins = margins[margins["exposure"].isin(STATE_LABELS.values())].copy()
absolute_rows = []
for origin, destinations, label in KEY_TRANSITIONS:
    risk = data[data["functional_state"].eq(origin)].copy()
    risk["event"] = risk["destination_state"].isin(destinations).astype(int)
    for code, exposure in STATE_LABELS.items():
        group = risk[risk["state"].eq(code)]
        margin = margins[(margins["transition"].eq(label)) & (margins["exposure"].eq(exposure))].iloc[0]
        absolute_rows.append({
            "transition": label,
            "exposure": exposure,
            "events": int(group["event"].sum()),
            "intervals": int(len(group)),
            "crude_event_rate": float(group["event"].mean()),
            "standardized_risk": float(margin["adjusted_risk"]),
            "standardized_ci_low": float(margin["risk_ci_low"]),
            "standardized_ci_high": float(margin["risk_ci_high"]),
        })

absolute_table = pd.DataFrame(absolute_rows)
absolute_table.to_csv(OUT / "table_complete_absolute_risks.csv", index=False, encoding="utf-8-sig")

qa = {
    "input_intervals": int(len(data)),
    "unique_people": int(data["ID"].nunique()),
    "duplicate_person_wave": int(data.duplicated(["ID", "wave"]).sum()),
    "sleep_observed": int(data["night_sleep_hours"].notna().sum()),
    "sleep_missing": int(data["night_sleep_hours"].isna().sum()),
    "sleep_category_counts": {str(k): int(v) for k, v in data["sleep_category"].value_counts().sort_index().items()},
    "sleep_models": int(len(sleep_table)),
    "absolute_risk_rows": int(len(absolute_table)),
    "all_sleep_models_finite": bool(np.isfinite(sleep_table["sleep_adjusted_estimate"]).all()),
    "all_standardized_risks_valid": bool(absolute_table["standardized_risk"].between(0, 1).all()),
}
(OUT / "sleep_and_absolute_risk_qa.json").write_text(
    json.dumps(qa, indent=2), encoding="utf-8"
)
print(json.dumps(qa, indent=2))
print(sleep_table.to_string(index=False))

