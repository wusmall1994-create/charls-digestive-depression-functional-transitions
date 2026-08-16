from pathlib import Path
import os
import json
import sys


import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["CHARLS_DATA_DIR"]).expanduser().resolve()
OUT = PROJECT / "outputs" / "comorbidity_state"
HARMONIZED = DATA / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta"
HSF2020 = DATA / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta"
SAMPLE2020 = DATA / "2020" / "CHARLS2020r" / "Sample_Infor.dta"

FUNCTION_LABELS = {
    0: "No limitation",
    1: "IADL limitation only",
    2: "ADL limitation",
    3: "Death",
}
WAVE_YEAR = {2: 2013, 3: 2015, 4: 2018, 5: 2020}

long = pd.read_csv(OUT / "comorbidity_state_long.csv", dtype={"ID": str})
baseline_ids = set(long.loc[long["wave"].eq(1), "ID"])
fixed_cols = [
    "ID", "communityID", "baseline_age", "female", "education_code", "marital_code",
    "rural_hukou", "current_smoker", "drank_last_year", "baseline_bmi",
    "log_hh_income", "other_chronic_count",
]
fixed = long.loc[long["wave"].eq(1), fixed_cols].drop_duplicates("ID")

hcols = ["ID"]
for wave in [2, 3, 4]:
    hcols.extend([
        f"inw{wave}", f"r{wave}iwstat", f"r{wave}adlab_c", f"r{wave}adlabm_c",
        f"r{wave}iadlza", f"r{wave}iadlzam",
    ])
h = pd.read_stata(HARMONIZED, columns=hcols, convert_categoricals=False)
h = h[h["ID"].isin(baseline_ids)].copy()

functional_frames = []
for wave in [2, 3, 4]:
    adl_ok = h[f"r{wave}adlabm_c"].eq(0)
    iadl_ok = h[f"r{wave}iadlzam"].eq(0)
    adl = h[f"r{wave}adlab_c"].gt(0).astype(float).where(adl_ok)
    iadl = h[f"r{wave}iadlza"].gt(0).astype(float).where(iadl_ok)
    fstate = pd.Series(np.nan, index=h.index, dtype="float64")
    ok = h[f"inw{wave}"].eq(1) & adl.notna() & iadl.notna()
    fstate.loc[ok & adl.eq(0) & iadl.eq(0)] = 0
    fstate.loc[ok & adl.eq(0) & iadl.eq(1)] = 1
    fstate.loc[ok & adl.eq(1)] = 2
    functional_frames.append(pd.DataFrame({
        "ID": h["ID"], "wave": wave, "year": WAVE_YEAR[wave],
        "alive_interview": h[f"inw{wave}"].eq(1).astype(int),
        "adl_limitation": adl, "iadl_limitation": iadl, "functional_state": fstate,
    }))

# Reconstruct the exact five-item Harmonized-IADL definition in 2020:
# phone, managing money, taking medication, shopping, and preparing meals.
adl20_items = ["db001", "db003", "db005", "db007", "db009", "db011"]
iadl20_items = ["db014", "db016", "db018", "db020", "db022"]
hsf20 = pd.read_stata(
    HSF2020, columns=["ID"] + adl20_items + iadl20_items, convert_categoricals=False
)
s20 = pd.read_stata(SAMPLE2020, convert_categoricals=False)
x20 = s20.merge(hsf20, on="ID", how="left")
x20 = x20[x20["ID"].isin(baseline_ids)].copy()
adl20_ok = x20[adl20_items].isin([1, 2, 3, 4]).all(axis=1)
iadl20_ok = x20[iadl20_items].isin([1, 2, 3, 4]).all(axis=1)
adl20 = x20[adl20_items].gt(1).any(axis=1).astype(float).where(adl20_ok)
iadl20 = x20[iadl20_items].gt(1).any(axis=1).astype(float).where(iadl20_ok)
fstate20 = pd.Series(np.nan, index=x20.index, dtype="float64")
ok20 = x20["died"].eq(0) & adl20.notna() & iadl20.notna()
fstate20.loc[ok20 & adl20.eq(0) & iadl20.eq(0)] = 0
fstate20.loc[ok20 & adl20.eq(0) & iadl20.eq(1)] = 1
fstate20.loc[ok20 & adl20.eq(1)] = 2
functional_frames.append(pd.DataFrame({
    "ID": x20["ID"], "wave": 5, "year": 2020,
    "alive_interview": x20["died"].eq(0).astype(int),
    "adl_limitation": adl20, "iadl_limitation": iadl20, "functional_state": fstate20,
}))

functional = pd.concat(functional_frames, ignore_index=True)
exposure = long[["ID", "wave", "state", "state_label", "digestive", "depression", "cesd10", "digestive_dispute"]]
functional = functional.merge(exposure, on=["ID", "wave"], how="left").merge(fixed, on="ID", how="left")
functional["functional_label"] = functional["functional_state"].map(FUNCTION_LABELS)
functional.to_csv(OUT / "functional_multistate_long.csv", index=False, encoding="utf-8-sig")

# One row per observed origin state and subsequent survey interval. Destination is
# a living functional state or death; unknown vital/functional status is censored.
rows = []
for start_wave, end_wave in [(2, 3), (3, 4), (4, 5)]:
    start = functional[functional["wave"].eq(start_wave)].copy()
    end = functional[functional["wave"].eq(end_wave)].set_index("ID")
    if end_wave in [3, 4]:
        vital = h.set_index("ID")[f"r{end_wave}iwstat"]
        death = start["ID"].map(vital).eq(5)
        known_alive = start["ID"].map(vital).isin([1, 4])
    else:
        vital = s20.set_index("ID")["died"]
        death = start["ID"].map(vital).eq(1)
        known_alive = start["ID"].map(vital).eq(0)
    destination_living = start["ID"].map(end["functional_state"])
    destination = pd.Series(np.nan, index=start.index, dtype="float64")
    destination.loc[death] = 3
    destination.loc[known_alive & destination_living.notna()] = destination_living.loc[
        known_alive & destination_living.notna()
    ]
    start["destination_state"] = destination
    start["outcome_observed"] = destination.notna().astype(int)
    start["known_dead"] = death.astype(int)
    start["interval"] = f"{WAVE_YEAR[start_wave]}-{WAVE_YEAR[end_wave]}"
    start["interval_years"] = WAVE_YEAR[end_wave] - WAVE_YEAR[start_wave]
    start["origin_label"] = start["functional_state"].map(FUNCTION_LABELS)
    start["destination_label"] = start["destination_state"].map(FUNCTION_LABELS)
    rows.append(start)

transitions = pd.concat(rows, ignore_index=True)
transitions = transitions[
    transitions["alive_interview"].eq(1) & transitions["functional_state"].notna() &
    transitions["state"].notna()
].copy()
transitions.to_csv(OUT / "functional_multistate_intervals_all.csv", index=False, encoding="utf-8-sig")
observed = transitions[transitions["outcome_observed"].eq(1)].copy()
observed.to_csv(OUT / "functional_multistate_intervals_observed.csv", index=False, encoding="utf-8-sig")

counts = (
    observed.groupby(["functional_state", "destination_state"], observed=True)
    .size().rename("n").reset_index()
)
counts["origin_label"] = counts["functional_state"].map(FUNCTION_LABELS)
counts["destination_label"] = counts["destination_state"].map(FUNCTION_LABELS)
counts["row_percent"] = counts["n"] / counts.groupby("functional_state")["n"].transform("sum") * 100
counts.to_csv(OUT / "table_functional_transition_counts.csv", index=False, encoding="utf-8-sig")

audit = {
    "functional_state_observed": {
        str(year): int(functional.loc[functional["year"].eq(year), "functional_state"].notna().sum())
        for year in [2013, 2015, 2018, 2020]
    },
    "eligible_origin_intervals": int(len(transitions)),
    "observed_destination_intervals": int(len(observed)),
    "censored_unknown_destination": int((transitions["outcome_observed"].eq(0)).sum()),
    "unique_people_observed_transitions": int(observed["ID"].nunique()),
    "deaths": int(observed["destination_state"].eq(3).sum()),
    "transition_counts": {
        f"{int(row.functional_state)}->{int(row.destination_state)}": int(row.n)
        for row in counts.itertuples()
    },
}
(OUT / "functional_multistate_build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))

