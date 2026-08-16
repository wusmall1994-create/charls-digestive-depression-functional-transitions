from __future__ import annotations

import json
import sys
from pathlib import Path
import os

PROJECT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2


DATA = Path(os.environ["CHARLS_DATA_DIR"]).expanduser().resolve()
OUT = PROJECT / "outputs" / "grip_discovery"
OUT.mkdir(parents=True, exist_ok=True)

HARMONIZED = DATA / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta"
BIO = {
    2011: DATA / "2011" / "household_and_community_questionnaire_data" / "biomarkers.dta",
    2013: DATA / "2013" / "CHARLS2013_Dataset" / "Biomarker.dta",
    2015: DATA / "2015" / "CHARLS2015r" / "Biomarker.dta",
}
HSF2020 = DATA / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta"
SAMPLE2020 = DATA / "2020" / "CHARLS2020r" / "Sample_Infor.dta"

CONDITIONS = [
    "hibpe", "dyslipe", "diabe", "cancre", "lunge", "livere", "hearte",
    "stroke", "kidneye", "psyche", "memrye", "arthre", "asthmae",
]
GRIP_COLS = ["qc003", "qc004", "qc005", "qc006"]
BADL20 = ["db001", "db003", "db005", "db007", "db009", "db011"]
IADL20 = ["db012", "db014", "db016", "db018", "db020", "db022"]


def valid_binary(s: pd.Series) -> pd.Series:
    return s.where(s.isin([0, 1]))


def build_grip(harm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_w1 = harm[["ID", "ID_w1"]].dropna(subset=["ID_w1"]).drop_duplicates("ID_w1")
    frames = []
    completion = []
    for year, path in BIO.items():
        raw = pd.read_stata(path, columns=["ID"] + GRIP_COLS, convert_categoricals=False)
        x = raw[GRIP_COLS].apply(pd.to_numeric, errors="coerce")
        x = x.where((x > 0) & (x <= 90))
        raw["valid_readings"] = x.notna().sum(axis=1)
        raw["grip_kg"] = x.max(axis=1)
        raw.loc[raw["valid_readings"] < 2, "grip_kg"] = np.nan
        raw["strict_four_readings"] = raw["valid_readings"].eq(4)
        if year == 2011:
            raw = raw.merge(key_w1, left_on="ID", right_on="ID_w1", how="left", validate="one_to_one")
            raw["person_id"] = raw["ID_y"]
        else:
            raw["person_id"] = raw["ID"]
        completion.append({
            "year": year,
            "biomarker_rows": len(raw),
            "at_least_2_valid": int(raw["grip_kg"].notna().sum()),
            "strict_4_valid": int(raw["strict_four_readings"].sum()),
        })
        frames.append(raw[["person_id", "grip_kg", "valid_readings", "strict_four_readings"]].assign(year=year))
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot(index="person_id", columns="year", values="grip_kg")
    wide.columns = [f"grip_{c}" for c in wide.columns]
    strict = long.pivot(index="person_id", columns="year", values="strict_four_readings")
    strict.columns = [f"strict_{c}" for c in strict.columns]
    result = wide.join(strict).reset_index()
    completion.append({
        "year": "2011-2015 intersection",
        "biomarker_rows": int(result[["grip_2011", "grip_2013", "grip_2015"]].notna().all(axis=1).sum()),
        "at_least_2_valid": int(result[["grip_2011", "grip_2013", "grip_2015"]].notna().all(axis=1).sum()),
        "strict_4_valid": int(result[["strict_2011", "strict_2013", "strict_2015"]].fillna(False).all(axis=1).sum()),
    })
    return result, pd.DataFrame(completion)


def build_2020_outcomes() -> pd.DataFrame:
    raw = pd.read_stata(HSF2020, columns=["ID"] + BADL20 + IADL20, convert_categoricals=False)
    for cols, stem in [(BADL20, "badl"), (IADL20, "iadl")]:
        items = raw[cols].apply(pd.to_numeric, errors="coerce").where(lambda d: d.isin([1, 2, 3, 4]))
        raw[f"{stem}20_observed_n"] = items.notna().sum(axis=1)
        raw[f"{stem}20_count"] = items.gt(1).sum(axis=1).where(items.notna().all(axis=1))
        raw[f"{stem}20_any"] = raw[f"{stem}20_count"].gt(0).astype(float)
        raw.loc[raw[f"{stem}20_count"].isna(), f"{stem}20_any"] = np.nan
    status = pd.read_stata(SAMPLE2020, columns=["ID", "died"], convert_categoricals=False)
    status["death_by_2020"] = pd.to_numeric(status["died"], errors="coerce").map({0: 0.0, 1: 1.0})
    return raw[["ID", "badl20_count", "badl20_any", "iadl20_count", "iadl20_any"]].rename(
        columns={"ID": "person_id"}
    ).merge(status[["ID", "death_by_2020"]].rename(columns={"ID": "person_id"}), on="person_id", how="outer")


def add_trajectories(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    grip_cols = ["grip_2011", "grip_2013", "grip_2015"]
    eligible = df[grip_cols + ["female"]].notna().all(axis=1)
    # Use the 2011 sex-specific distribution as a fixed reference for all waves.
    # Wave-specific standardization would erase the cohort-wide longitudinal decline.
    for col in grip_cols:
        df[f"z_{col}"] = np.nan
        for sex in [0.0, 1.0]:
            ix = eligible & df["female"].eq(sex)
            ref_mean = df.loc[ix, "grip_2011"].mean()
            ref_sd = df.loc[ix, "grip_2011"].std(ddof=0)
            df.loc[ix, f"z_{col}"] = (df.loc[ix, col] - ref_mean) / ref_sd
    zcols = [f"z_{c}" for c in grip_cols]
    X = df.loc[eligible, zcols].to_numpy()
    choices = []
    fitted = {}
    for k in range(2, 5):
        model = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=5,
            max_iter=500, random_state=20260812,
        )
        labels = model.fit_predict(X)
        counts = np.bincount(labels, minlength=k)
        choices.append({
            "classes": k, "bic": model.bic(X), "aic": model.aic(X),
            "smallest_class_n": int(counts.min()), "smallest_class_pct": float(counts.min() / len(X)),
        })
        fitted[k] = (model, labels)
    viable = [x for x in choices if x["smallest_class_pct"] >= 0.05]
    selected = min(viable or choices, key=lambda x: x["bic"])["classes"]
    labels = fitted[selected][1]
    tmp = pd.DataFrame(X, columns=zcols).assign(raw_class=labels)
    profile = tmp.groupby("raw_class")[zcols].mean()
    profile["mean_level"] = profile.mean(axis=1)
    profile["change"] = profile[zcols[-1]] - profile[zcols[0]]
    order = profile.sort_values("mean_level").index.tolist()
    rank = {raw: i for i, raw in enumerate(order)}
    if selected == 2:
        names = ["Lower", "Higher"]
    elif selected == 3:
        names = ["Low", "Intermediate", "High"]
    else:
        names = ["Lowest", "Lower-middle", "Upper-middle", "Highest"]
    mapping = {raw: names[rank[raw]] for raw in order}
    df["trajectory"] = pd.NA
    df.loc[eligible, "trajectory"] = pd.Series(labels, index=df.index[eligible]).map(mapping)
    profile = profile.reset_index()
    profile["trajectory"] = profile["raw_class"].map(mapping)
    profile["n"] = profile["raw_class"].map(pd.Series(labels).value_counts())
    profile["pct"] = profile["n"] / len(labels)
    # Individual linear slope in kg/year; useful for a less model-dependent sensitivity analysis.
    years = np.array([0.0, 2.0, 4.0])
    y = df.loc[eligible, grip_cols].to_numpy()
    slopes = ((y - y.mean(axis=1, keepdims=True)) * (years - years.mean())).sum(axis=1) / ((years - years.mean()) ** 2).sum()
    df["grip_slope_kg_y"] = np.nan
    df.loc[eligible, "grip_slope_kg_y"] = slopes
    df["grip_slope_per_sd"] = (df["grip_slope_kg_y"] - df.loc[eligible, "grip_slope_kg_y"].mean()) / df.loc[eligible, "grip_slope_kg_y"].std(ddof=0)
    return df, pd.DataFrame(choices), selected, profile


def tidy_logit(model, model_name: str) -> pd.DataFrame:
    ci = model.conf_int()
    out = pd.DataFrame({
        "term": model.params.index,
        "estimate_log_or": model.params.values,
        "se": model.bse.values,
        "p_value": model.pvalues.values,
        "or": np.exp(model.params.values),
        "ci_low": np.exp(ci[0].values),
        "ci_high": np.exp(ci[1].values),
    })
    out.insert(0, "model", model_name)
    out["n"] = int(model.nobs)
    return out


def trajectory_stratum_contrasts(model, model_name: str) -> pd.DataFrame:
    rows = []
    cov = model.cov_params()
    for digestive in [0, 1]:
        for level in ["Lower-middle", "Lowest", "Upper-middle"]:
            main = f"C(trajectory)[T.{level}]"
            inter = f"C(trajectory)[T.{level}]:digestive_2011"
            if main not in model.params:
                continue
            beta = model.params[main]
            var = cov.loc[main, main]
            if digestive == 1 and inter in model.params:
                beta += model.params[inter]
                var += cov.loc[inter, inter] + 2 * cov.loc[main, inter]
            se = np.sqrt(max(var, 0))
            rows.append({
                "model": model_name, "digestive_2011": digestive,
                "contrast": f"{level} vs Highest", "or": np.exp(beta),
                "ci_low": np.exp(beta - 1.96 * se), "ci_high": np.exp(beta + 1.96 * se),
                "p_value": 2 * (1 - __import__("scipy").stats.norm.cdf(abs(beta / se))) if se > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def fit_logit(data: pd.DataFrame, outcome: str, exposure: str, interaction: bool, add_mediator: bool, label: str, weighted: bool = False):
    cov = "age_2011 + female + C(education) + partnered_2011 + rural_2011 + smoker_2011 + drinker_2011 + bmi_2011 + baseline_comorbidity_count"
    if interaction:
        exp = f"C({exposure}) * digestive_2011" if exposure == "trajectory" else f"{exposure} * digestive_2011"
    else:
        exp = exposure
    mediator = " + comorbidity_count_2018" if add_mediator else ""
    formula = f"{outcome} ~ {exp} + {cov}{mediator}"
    required = [outcome, exposure, "digestive_2011", "age_2011", "female", "education", "partnered_2011", "rural_2011", "smoker_2011", "drinker_2011", "bmi_2011", "baseline_comorbidity_count"]
    if add_mediator:
        required.append("comorbidity_count_2018")
    d = data.dropna(subset=required).copy()
    weight = None
    if weighted:
        d = d.loc[d["weight_2018"].notna() & d["weight_2018"].gt(0)].copy()
        weight = d["weight_2018"] / d["weight_2018"].mean()
    model = smf.glm(formula, data=d, family=sm.families.Binomial(), freq_weights=weight).fit(cov_type="HC1")
    return model, tidy_logit(model, label), formula, len(d), int(d[outcome].sum())


def main():
    hcols = ["ID", "ID_w1", "ragender", "raeducl"]
    common_stems = ["agey", "mstat", "rural2", "smoken", "drinkl", "digeste", "adlab_c", "wtrespb"]
    for w in [1, 4]:
        hcols.extend([f"r{w}{s}" for s in common_stems])
        hcols.extend([f"r{w}{s}" for s in CONDITIONS])
    hcols.extend(["r1mbmi", "r4iadlza"])
    harm = pd.read_stata(HARMONIZED, columns=hcols, convert_categoricals=False)
    df = pd.DataFrame({
        "person_id": harm["ID"],
        "age_2011": harm["r1agey"],
        "female": harm["ragender"].map({1.0: 0.0, 2.0: 1.0}),
        "education": harm["raeducl"].map({1.0: "Less than upper secondary", 2.0: "Upper secondary/vocational", 3.0: "Tertiary"}),
        "partnered_2011": harm["r1mstat"].isin([1.0, 3.0]).astype(float),
        "rural_2011": valid_binary(harm["r1rural2"]),
        "smoker_2011": valid_binary(harm["r1smoken"]),
        "drinker_2011": valid_binary(harm["r1drinkl"]),
        "bmi_2011": harm["r1mbmi"].where(harm["r1mbmi"].between(10, 60)),
        "digestive_2011": valid_binary(harm["r1digeste"]),
        "badl_2011": harm["r1adlab_c"].gt(0).astype(float),
        "badl_2018": harm["r4adlab_c"].gt(0).astype(float),
        "iadl_2018": harm["r4iadlza"].gt(0).astype(float),
        "weight_2018": harm["r4wtrespb"],
    })
    for col, rawcol in [("partnered_2011", "r1mstat"), ("badl_2011", "r1adlab_c"), ("badl_2018", "r4adlab_c"), ("iadl_2018", "r4iadlza")]:
        df.loc[harm[rawcol].isna(), col] = np.nan
    for w, suffix in [(1, "baseline"), (4, "2018")]:
        cond = pd.DataFrame({c: valid_binary(harm[f"r{w}{c}"]) for c in CONDITIONS})
        df[f"{suffix}_comorbidity_observed_n"] = cond.notna().sum(axis=1)
        df[f"{suffix}_comorbidity_count"] = cond.sum(axis=1).where(cond.notna().all(axis=1))
    df = df.rename(columns={"baseline_comorbidity_count": "baseline_comorbidity_count", "2018_comorbidity_count": "comorbidity_count_2018"})

    grip, completion = build_grip(harm)
    out20 = build_2020_outcomes()
    df = df.merge(grip, on="person_id", how="left", validate="one_to_one").merge(out20, on="person_id", how="left", validate="one_to_one")
    df, selection, selected_k, profiles = add_trajectories(df)

    # Incident outcomes require no corresponding limitation in 2018. The primary
    # cohort also excludes baseline BADL to reduce reverse causation during the exposure window.
    df["incident_badl_2020"] = df["badl20_any"].where(df["badl_2018"].eq(0))
    df["incident_iadl_2020"] = df["iadl20_any"].where(df["iadl_2018"].eq(0))
    df["primary_eligible"] = (
        df["age_2011"].ge(45) & df["badl_2011"].eq(0)
        & df[["grip_2011", "grip_2013", "grip_2015"]].notna().all(axis=1)
    )
    analysis = df.loc[df["primary_eligible"]].copy()

    flow = []
    masks = [
        ("Harmonized CHARLS respondents", pd.Series(True, index=df.index)),
        ("Age >=45 in 2011", df["age_2011"].ge(45)),
        ("Grip observed in 2011", df["age_2011"].ge(45) & df["grip_2011"].notna()),
        ("Grip observed in 2011, 2013, 2015", df["age_2011"].ge(45) & df[["grip_2011", "grip_2013", "grip_2015"]].notna().all(axis=1)),
        ("Also no BADL in 2011 (primary exposure cohort)", df["primary_eligible"]),
        ("Also no IADL in 2018 and observed in 2020", df["primary_eligible"] & df["iadl_2018"].eq(0) & df["iadl20_any"].notna()),
        ("Complete covariates for IADL model", df["primary_eligible"] & df["iadl_2018"].eq(0) & df["iadl20_any"].notna() & df[["digestive_2011", "age_2011", "female", "education", "partnered_2011", "rural_2011", "smoker_2011", "drinker_2011", "bmi_2011", "baseline_comorbidity_count"]].notna().all(axis=1)),
    ]
    previous = None
    for step, mask in masks:
        n = int(mask.sum())
        flow.append({"step": step, "n": n, "retained_from_previous": n / previous if previous else np.nan})
        previous = n

    models = []
    model_meta = []
    fitted_models = {}
    contrasts = []
    for outcome in ["incident_iadl_2020", "incident_badl_2020"]:
        for exposure, inter, med, label in [
            ("trajectory", True, False, f"{outcome}: trajectory x digestive"),
            ("trajectory", True, True, f"{outcome}: trajectory x digestive + mediator"),
            ("grip_slope_per_sd", True, False, f"{outcome}: slope x digestive"),
            ("grip_slope_per_sd", True, True, f"{outcome}: slope x digestive + mediator"),
        ]:
            try:
                model, tidy, formula, n, events = fit_logit(analysis, outcome, exposure, inter, med, label)
                models.append(tidy)
                fitted_models[label] = model
                if exposure == "trajectory":
                    contrasts.append(trajectory_stratum_contrasts(model, label))
                model_meta.append({"model": label, "formula": formula, "n": n, "events": events, "converged": bool(model.converged)})
            except Exception as exc:
                model_meta.append({"model": label, "formula": "", "n": None, "events": None, "converged": False, "error": repr(exc)})

    # Sampling-weighted sensitivity models for the primary trajectory exposure.
    for outcome in ["incident_iadl_2020", "incident_badl_2020"]:
        label = f"{outcome}: trajectory x digestive, survey weighted"
        try:
            model, tidy, formula, n, events = fit_logit(analysis, outcome, "trajectory", True, False, label, weighted=True)
            models.append(tidy)
            fitted_models[label] = model
            contrasts.append(trajectory_stratum_contrasts(model, label))
            model_meta.append({"model": label, "formula": formula, "n": n, "events": events, "converged": bool(model.converged)})
        except Exception as exc:
            model_meta.append({"model": label, "formula": "", "n": None, "events": None, "converged": False, "error": repr(exc)})

    # Strict measurement sensitivity: all four grip readings valid at every exposure wave.
    strict_analysis = analysis.loc[analysis[["strict_2011", "strict_2013", "strict_2015"]].fillna(False).all(axis=1)].copy()
    for outcome in ["incident_iadl_2020", "incident_badl_2020"]:
        label = f"{outcome}: trajectory x digestive, strict 4 readings"
        try:
            model, tidy, formula, n, events = fit_logit(strict_analysis, outcome, "trajectory", True, False, label)
            models.append(tidy)
            fitted_models[label] = model
            contrasts.append(trajectory_stratum_contrasts(model, label))
            model_meta.append({"model": label, "formula": formula, "n": n, "events": events, "converged": bool(model.converged)})
        except Exception as exc:
            model_meta.append({"model": label, "formula": "", "n": None, "events": None, "converged": False, "error": repr(exc)})

    global_rows = []
    for label, full in fitted_models.items():
        if "x digestive" not in label or "mediator" in label:
            continue
        outcome = label.split(":")[0]
        exposure = "trajectory" if "trajectory" in label else "grip_slope_per_sd"
        weighted = "survey weighted" in label
        cov = "age_2011 + female + C(education) + partnered_2011 + rural_2011 + smoker_2011 + drinker_2011 + bmi_2011 + baseline_comorbidity_count"
        main_exp = f"C({exposure})" if exposure == "trajectory" else exposure
        formula0 = f"{outcome} ~ {main_exp} + digestive_2011 + {cov}"
        base_for_test = strict_analysis if "strict 4 readings" in label else analysis
        d0 = base_for_test.dropna(subset=[outcome, exposure, "digestive_2011", "age_2011", "female", "education", "partnered_2011", "rural_2011", "smoker_2011", "drinker_2011", "bmi_2011", "baseline_comorbidity_count"]).copy()
        wt = None
        if weighted:
            d0 = d0.loc[d0["weight_2018"].notna() & d0["weight_2018"].gt(0)].copy()
            wt = d0["weight_2018"] / d0["weight_2018"].mean()
        reduced = smf.glm(formula0, data=d0, family=sm.families.Binomial(), freq_weights=wt).fit()
        lr = 2 * (full.llf - reduced.llf)
        ddf = int(full.df_model - reduced.df_model)
        global_rows.append({"model": label, "test": "Likelihood-ratio test for all interaction terms", "chi2": lr, "df": ddf, "p_value": chi2.sf(max(lr, 0), ddf)})

    # Exploratory mediation on continuous grip slope. OLS for disease count and
    # logistic outcome yield a product-of-coefficients estimate on the log-odds scale.
    med_rows = []
    cov = "age_2011 + female + C(education) + partnered_2011 + rural_2011 + smoker_2011 + drinker_2011 + bmi_2011 + baseline_comorbidity_count"
    for digestive in [0.0, 1.0]:
        d = analysis.loc[analysis["digestive_2011"].eq(digestive)].dropna(subset=["incident_iadl_2020", "grip_slope_per_sd", "comorbidity_count_2018", "age_2011", "female", "education", "partnered_2011", "rural_2011", "smoker_2011", "drinker_2011", "bmi_2011", "baseline_comorbidity_count"]).copy()
        if len(d) < 200 or d["incident_iadl_2020"].sum() < 20:
            continue
        mm = smf.ols(f"comorbidity_count_2018 ~ grip_slope_per_sd + {cov}", data=d).fit(cov_type="HC1")
        yy = smf.glm(f"incident_iadl_2020 ~ grip_slope_per_sd + comorbidity_count_2018 + {cov}", data=d, family=sm.families.Binomial()).fit(cov_type="HC1")
        a, b = mm.params["grip_slope_per_sd"], yy.params["comorbidity_count_2018"]
        se_prod = np.sqrt((b * mm.bse["grip_slope_per_sd"]) ** 2 + (a * yy.bse["comorbidity_count_2018"]) ** 2)
        med_rows.append({
            "digestive_2011": int(digestive), "n": len(d), "iadl_events": int(d["incident_iadl_2020"].sum()),
            "a_slope_to_comorbidity": a, "b_comorbidity_to_logodds": b,
            "indirect_logodds": a * b, "indirect_se_sobel": se_prod,
            "indirect_ci_low": a * b - 1.96 * se_prod, "indirect_ci_high": a * b + 1.96 * se_prod,
            "note": "Exploratory product-of-coefficients; not a causal mediation estimate.",
        })

    desc = analysis.groupby(["trajectory", "digestive_2011"], observed=True).agg(
        n=("person_id", "size"), age_mean=("age_2011", "mean"), female_pct=("female", "mean"),
        grip_2011_mean=("grip_2011", "mean"), grip_2015_mean=("grip_2015", "mean"),
        comorbidity_2018_mean=("comorbidity_count_2018", "mean"),
        iadl_risk=("incident_iadl_2020", "mean"), badl_risk=("incident_badl_2020", "mean"),
    ).reset_index()

    df.to_csv(OUT / "charls_grip_discovery_person_level.csv", index=False, encoding="utf-8-sig")
    completion.to_csv(OUT / "grip_completion.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flow).to_csv(OUT / "cohort_flow.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(OUT / "trajectory_model_selection.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(OUT / "trajectory_profiles.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUT / "descriptive_by_trajectory_digestive.csv", index=False, encoding="utf-8-sig")
    pd.concat(models, ignore_index=True).to_csv(OUT / "regression_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(model_meta).to_csv(OUT / "model_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(global_rows).to_csv(OUT / "interaction_global_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat(contrasts, ignore_index=True).to_csv(OUT / "trajectory_stratum_contrasts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(med_rows).to_csv(OUT / "exploratory_mediation.csv", index=False, encoding="utf-8-sig")

    audit = {
        "selected_trajectory_classes": int(selected_k),
        "primary_exposure_cohort_n": int(len(analysis)),
        "digestive_prevalence": float(analysis["digestive_2011"].mean()),
        "iadl_observed_n": int(analysis["incident_iadl_2020"].notna().sum()),
        "iadl_events": int(analysis["incident_iadl_2020"].sum(skipna=True)),
        "badl_observed_n": int(analysis["incident_badl_2020"].notna().sum()),
        "badl_events": int(analysis["incident_badl_2020"].sum(skipna=True)),
        "notes": [
            "Grip is the maximum of up to four readings; at least two valid readings (>0 and <=90 kg) are required per wave.",
            "Trajectory inputs use the 2011 sex-specific mean and SD as a fixed reference for all waves; 2-4 diagonal Gaussian mixture classes were compared by BIC with a 5% minimum class rule.",
            "2018 BADL/IADL uses Harmonized CHARLS variables that resolve survey skip patterns; 2020 outcomes use all six raw items.",
            "2018 comorbidity count excludes digestive disease to avoid mechanical overlap with the modifier.",
            "Primary exposure cohort excludes baseline BADL; outcome-specific models additionally require no corresponding 2018 disability.",
            "Mediation output is exploratory and should not be interpreted causally without stronger identification and attrition handling.",
        ],
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

