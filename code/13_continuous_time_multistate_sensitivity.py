from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

SOURCE = ROOT / "outputs" / "comorbidity_state" / "functional_multistate_intervals_weighted.csv"
OUT = ROOT / "outputs" / "digestive_depression_multistate"
EDGES = [(i, j) for i in range(3) for j in range(4) if i != j]
KEY = {
    "No limitation to ADL limitation": [(0, 2)],
    "IADL limitation to ADL limitation": [(1, 2)],
    "IADL limitation to no limitation (recovery)": [(1, 0)],
    "ADL limitation to functional improvement": [(2, 0), (2, 1)],
}


def qmatrix(theta):
    q = np.zeros((4, 4), dtype=float)
    rates = np.exp(np.clip(theta, -10, 3))
    for rate, (i, j) in zip(rates, EDGES):
        q[i, j] = rate
    for i in range(3):
        q[i, i] = -q[i].sum()
    return q


def aggregate(x):
    return x.groupby(["functional_state", "destination_state", "interval_years"], observed=True)["iow"].sum().reset_index(name="weight")


def initial(x):
    exposure = x.groupby("functional_state").apply(lambda g: float((g.iow * g.interval_years).sum()), include_groups=False).to_dict()
    vals = []
    for i, j in EDGES:
        events = float(x.loc[x.functional_state.eq(i) & x.destination_state.eq(j), "iow"].sum())
        vals.append(np.log(max(events / max(exposure.get(i, 1), 1), 0.002)))
    return np.asarray(vals)


def fit_group(x, start=None):
    a = aggregate(x)
    def objective(theta):
        q = qmatrix(theta)
        mats = {t: expm(q * t) for t in a.interval_years.unique()}
        ll = 0.0
        for row in a.itertuples():
            p = mats[row.interval_years][int(row.functional_state), int(row.destination_state)]
            ll += row.weight * np.log(max(p, 1e-12))
        return -ll
    if start is None:
        start = initial(x)
    fit = minimize(objective, start, method="L-BFGS-B", bounds=[(-10, 3)] * len(EDGES), options={"maxiter": 1000, "ftol": 1e-10})
    return fit, qmatrix(fit.x)


def key_rates(q):
    return {label: sum(q[i, j] for i, j in edges) for label, edges in KEY.items()}


df = pd.read_csv(SOURCE, dtype={"ID": str})
df = df[df.state.isin([2, 3])].copy()
fits, qs = {}, {}
for state in [2, 3]:
    fits[state], qs[state] = fit_group(df[df.state.eq(state)])

base_rates, co_rates = key_rates(qs[2]), key_rates(qs[3])
rows = []
for label in KEY:
    rows.append({"transition": label, "intensity_depression_only_per_year": base_rates[label],
                 "intensity_cooccurring_per_year": co_rates[label],
                 "intensity_ratio_cooccurring_vs_depression_only": co_rates[label] / base_rates[label]})
results = pd.DataFrame(rows)

# Participant bootstrap preserves within-person interval dependence.
rng = np.random.default_rng(20260813)
boot = []
ids_by_state = {s: df.loc[df.state.eq(s), "ID"].unique() for s in [2, 3]}
for b in range(50):
    qbs = {}
    ok = True
    for state in [2, 3]:
        ids = ids_by_state[state]
        sampled = rng.choice(ids, size=len(ids), replace=True)
        multiplicity = pd.Series(sampled).value_counts().rename("mult")
        xb = df[df.state.eq(state)].merge(multiplicity, left_on="ID", right_index=True, how="inner")
        xb["iow"] = xb.iow * xb.mult
        fb, qb = fit_group(xb, fits[state].x)
        if not fb.success:
            ok = False
            break
        qbs[state] = qb
    if ok:
        r2, r3 = key_rates(qbs[2]), key_rates(qbs[3])
        boot.append({"replicate": b + 1, **{label: r3[label] / r2[label] for label in KEY}})
boot = pd.DataFrame(boot)
for i, label in enumerate(KEY):
    if len(boot):
        results.loc[i, "bootstrap_ci_low"] = boot[label].quantile(.025)
        results.loc[i, "bootstrap_ci_high"] = boot[label].quantile(.975)

results.to_csv(OUT / "table_continuous_time_multistate.csv", index=False, encoding="utf-8-sig")
boot.to_csv(OUT / "continuous_time_multistate_bootstrap.csv", index=False, encoding="utf-8-sig")
qa = {
    "model": "Panel-observed continuous-time homogeneous Markov model with death absorbing and all transitions among the three living states allowed.",
    "exposure": "Separate transition-intensity matrices for depression-only and co-occurring origin intervals; origin exposure assumed constant within each interval.",
    "intervals": int(len(df)), "people": int(df.ID.nunique()),
    "depression_only_intervals": int(df.state.eq(2).sum()), "cooccurring_intervals": int(df.state.eq(3).sum()),
    "both_models_converged": bool(fits[2].success and fits[3].success),
    "bootstrap_requested": 50, "bootstrap_successful": int(len(boot)),
    "feasible": bool(fits[2].success and fits[3].success and len(boot) >= 45),
    "estimand_boundary": "Sensitivity analysis without covariate adjustment; it evaluates whether unequal interval lengths and panel observation alter the transition pattern.",
}
(OUT / "continuous_time_multistate_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps(qa, indent=2))
print(results.to_string(index=False))

