from pathlib import Path
import os
import json
import sys


import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["CHARLS_DATA_DIR"]).expanduser().resolve()
OUT = PROJECT / "outputs" / "comorbidity_state"
OUT.mkdir(parents=True, exist_ok=True)

HARMONIZED = DATA / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta"
HSF2020 = DATA / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta"
SAMPLE2020 = DATA / "2020" / "CHARLS2020r" / "Sample_Infor.dta"

WAVES = {1: 2011, 2: 2013, 3: 2015, 4: 2018}
NEXT_WAVE = {1: 2, 2: 3, 3: 4}
STATE_LABELS = {
    0: "Neither",
    1: "Digestive only",
    2: "Depression only",
    3: "Co-occurring",
}

OTHER_CONDITIONS = [
    "hibpe", "dyslipe", "diabe", "cancre", "lunge", "livere",
    "hearte", "stroke", "kidneye", "psyche", "memrye", "arthre", "asthmae",
]


def numeric_binary(series):
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[series.eq(0)] = 0.0
    out.loc[series.eq(1)] = 1.0
    return out


def decimal_date(year, month, fallback):
    y = pd.to_numeric(year, errors="coerce")
    m = pd.to_numeric(month, errors="coerce")
    out = y + (m.clip(1, 12).fillna(7) - 0.5) / 12
    return out.fillna(fallback)


base_static = ["ID", "ragender", "raeducl", "communityID", "hh1itot"]
wave_stems = [
    "agey", "iwy", "iwm", "iwstat", "mstat", "rural2", "smoken", "drinkl",
    "mbmi", "cesd10", "cesd10m", "digeste", "digestf", "adlab_c", "adlabm_c",
    "wtrespb",
]
columns = base_static + [f"inw{w}" for w in WAVES]
for w in WAVES:
    for stem in wave_stems:
        candidate = f"r{w}{stem}"
        # Measured BMI is not available in harmonized wave 4.
        if candidate != "r4mbmi":
            columns.append(candidate)
    columns.extend(f"r{w}{stem}" for stem in OTHER_CONDITIONS)

wide = pd.read_stata(HARMONIZED, columns=columns, convert_categoricals=False)
baseline_ids = set(wide.loc[wide["inw1"].eq(1), "ID"])
wide = wide[wide["ID"].isin(baseline_ids)].copy()

# Fixed baseline confounders. Missing BMI/income are retained with explicit indicators
# in the analysis script; no outcome or exposure information is used for imputation.
wide["baseline_age"] = wide["r1agey"]
wide["female"] = numeric_binary(wide["ragender"] - 1)  # harmonized: 1 male, 2 female
wide["education_code"] = wide["raeducl"]
wide["marital_code"] = wide["r1mstat"]
wide["rural_hukou"] = numeric_binary(wide["r1rural2"])
wide["current_smoker"] = numeric_binary(wide["r1smoken"])
wide["drank_last_year"] = numeric_binary(wide["r1drinkl"])
wide["baseline_bmi"] = wide["r1mbmi"].where(wide["r1mbmi"].between(10, 60))
wide["baseline_hh_income"] = wide["hh1itot"].where(wide["hh1itot"].ge(0))
wide["log_hh_income"] = np.log1p(wide["baseline_hh_income"])
condition_cols = [f"r1{x}" for x in OTHER_CONDITIONS]
wide["other_chronic_count"] = wide[condition_cols].eq(1).sum(axis=1)

fixed_cols = [
    "ID", "communityID", "baseline_age", "female", "education_code", "marital_code",
    "rural_hukou", "current_smoker", "drank_last_year", "baseline_bmi",
    "log_hh_income", "other_chronic_count",
]
fixed = wide[fixed_cols].copy()

long_frames = []
for wave, year in WAVES.items():
    interviewed = wide[f"inw{wave}"].eq(1)
    cesd_complete = wide[f"r{wave}cesd10m"].eq(0)
    digestive = numeric_binary(wide[f"r{wave}digeste"])
    depression = pd.Series(np.nan, index=wide.index, dtype="float64")
    depression.loc[cesd_complete] = wide.loc[cesd_complete, f"r{wave}cesd10"].ge(10).astype(float)
    adl_complete = wide[f"r{wave}adlabm_c"].eq(0)
    adl_disability = pd.Series(np.nan, index=wide.index, dtype="float64")
    adl_disability.loc[adl_complete] = wide.loc[adl_complete, f"r{wave}adlab_c"].gt(0).astype(float)
    state = pd.Series(np.nan, index=wide.index, dtype="float64")
    state_ok = interviewed & digestive.notna() & depression.notna()
    state.loc[state_ok] = digestive.loc[state_ok] + 2 * depression.loc[state_ok]
    frame = pd.DataFrame({
        "ID": wide["ID"],
        "wave": wave,
        "year": year,
        "interviewed_alive": interviewed.astype(int),
        "interview_time": decimal_date(wide[f"r{wave}iwy"], wide[f"r{wave}iwm"], year + 0.5),
        "digestive": digestive,
        "digestive_dispute": wide[f"r{wave}digestf"].fillna(0).ne(0).astype(int),
        "cesd10": wide[f"r{wave}cesd10"].where(cesd_complete),
        "depression": depression,
        "state": state,
        "adl_disability": adl_disability,
        "wave_weight": wide[f"r{wave}wtrespb"],
    })
    long_frames.append(frame)

# Raw 2020 data allow a compatible disease status, CES-D10 and six-item ADL definition.
hsf20_cols = ["ID", "da002_10_", "da003_10_"] + [f"dc{i:03d}" for i in range(16, 26)] + [
    "db001", "db003", "db005", "db007", "db009", "db011"
]
hsf20 = pd.read_stata(HSF2020, columns=hsf20_cols, convert_categoricals=False)
sample20 = pd.read_stata(SAMPLE2020, convert_categoricals=False)
raw20 = sample20.merge(hsf20, on="ID", how="left", suffixes=("", "_hsf"))
raw20 = raw20[raw20["ID"].isin(baseline_ids)].copy()

digest20 = pd.Series(np.nan, index=raw20.index, dtype="float64")
digest20.loc[raw20["da002_10_"].isin([1, 2, 3])] = 1
digest20.loc[raw20["da002_10_"].eq(99)] = 0
digest20.loc[raw20["da003_10_"].eq(1)] = 1
digest20.loc[raw20["da003_10_"].eq(2)] = 0

cesd_items = [f"dc{i:03d}" for i in range(16, 26)]
valid_cesd = raw20[cesd_items].isin([1, 2, 3, 4]).all(axis=1)
scored = raw20[cesd_items] - 1
for positive in ["dc020", "dc023"]:
    scored[positive] = 3 - scored[positive]
cesd20 = scored.sum(axis=1).where(valid_cesd)
depression20 = cesd20.ge(10).astype(float).where(valid_cesd)

adl_items = ["db001", "db003", "db005", "db007", "db009", "db011"]
valid_adl = raw20[adl_items].isin([1, 2, 3, 4]).all(axis=1)
adl20 = raw20[adl_items].gt(1).any(axis=1).astype(float).where(valid_adl)
state20 = (digest20 + 2 * depression20).where(digest20.notna() & depression20.notna())

frame20 = pd.DataFrame({
    "ID": raw20["ID"],
    "wave": 5,
    "year": 2020,
    "interviewed_alive": raw20["died"].eq(0).astype(int),
    "interview_time": decimal_date(raw20["iyear"], raw20["imonth"], 2020.5),
    "digestive": digest20,
    "digestive_dispute": 0,
    "cesd10": cesd20,
    "depression": depression20,
    "state": state20,
    "adl_disability": adl20,
    "wave_weight": np.nan,
})
long = pd.concat(long_frames + [frame20], ignore_index=True).merge(fixed, on="ID", how="left")
long["state_label"] = long["state"].map(STATE_LABELS)
long.to_csv(OUT / "comorbidity_state_long.csv", index=False, encoding="utf-8-sig")

# Construct discrete mortality intervals. Harmonized interview status 5 means newly
# confirmed dead in that wave; status 1/4 means known alive. Unknown vital status is censored.
mortality_rows = []
for wave, next_wave in NEXT_WAVE.items():
    start = long[long["wave"].eq(wave)].copy()
    vital = wide.set_index("ID")[f"r{next_wave}iwstat"]
    start["next_vital"] = start["ID"].map(vital)
    start["death"] = np.where(start["next_vital"].eq(5), 1,
                              np.where(start["next_vital"].isin([1, 4]), 0, np.nan))
    start["interval"] = f"{WAVES[wave]}-{WAVES[next_wave]}"
    start["interval_years"] = WAVES[next_wave] - WAVES[wave]
    mortality_rows.append(start)

start18 = long[long["wave"].eq(4)].copy()
vital20 = sample20.set_index("ID")["died"]
start18["next_vital"] = start18["ID"].map(vital20)
start18["death"] = np.where(start18["next_vital"].eq(1), 1,
                            np.where(start18["next_vital"].eq(0), 0, np.nan))
start18["interval"] = "2018-2020"
start18["interval_years"] = 2
mortality_rows.append(start18)

mortality = pd.concat(mortality_rows, ignore_index=True)
mortality = mortality[
    mortality["interviewed_alive"].eq(1) & mortality["state"].notna() & mortality["death"].notna()
].copy()
mortality["death"] = mortality["death"].astype(int)
mortality.to_csv(OUT / "mortality_person_intervals.csv", index=False, encoding="utf-8-sig")

# First incident six-item ADL disability. Participants leave the risk set after the
# first observed incident event. Consecutive observed waves are required.
adl_rows = []
first_event_ids = set()
for wave in [1, 2, 3, 4]:
    a = long[long["wave"].eq(wave)].set_index("ID")
    b = long[long["wave"].eq(wave + 1)].set_index("ID")
    ids = a.index.intersection(b.index)
    pair = a.loc[ids].reset_index()
    pair["next_adl"] = pair["ID"].map(b["adl_disability"])
    pair["next_alive_interview"] = pair["ID"].map(b["interviewed_alive"])
    pair = pair[
        pair["interviewed_alive"].eq(1) & pair["state"].notna() &
        pair["adl_disability"].eq(0) & pair["next_alive_interview"].eq(1) &
        pair["next_adl"].notna() & ~pair["ID"].isin(first_event_ids)
    ].copy()
    pair["incident_adl"] = pair["next_adl"].astype(int)
    pair["interval"] = f"{int(a['year'].iloc[0])}-{int(b['year'].iloc[0])}"
    pair["interval_years"] = int(b["year"].iloc[0] - a["year"].iloc[0])
    first_event_ids.update(pair.loc[pair["incident_adl"].eq(1), "ID"])
    adl_rows.append(pair)
adl_incident = pd.concat(adl_rows, ignore_index=True)
adl_incident.to_csv(OUT / "incident_adl_person_intervals.csv", index=False, encoding="utf-8-sig")

# Consecutive-wave state transitions among living respondents with both states observed.
transition_rows = []
for wave in [1, 2, 3, 4]:
    a = long[long["wave"].eq(wave)].set_index("ID")
    b = long[long["wave"].eq(wave + 1)].set_index("ID")
    ids = a.index.intersection(b.index)
    tr = pd.DataFrame({
        "ID": ids,
        "from_state": a.loc[ids, "state"].values,
        "to_state": b.loc[ids, "state"].values,
        "to_interviewed_alive": b.loc[ids, "interviewed_alive"].values,
    })
    tr = tr[tr["from_state"].notna() & tr["to_state"].notna() & tr["to_interviewed_alive"].eq(1)].copy()
    tr["interval"] = f"{int(a['year'].iloc[0])}-{int(b['year'].iloc[0])}"
    transition_rows.append(tr)
transitions = pd.concat(transition_rows, ignore_index=True)
transitions["from_label"] = transitions["from_state"].map(STATE_LABELS)
transitions["to_label"] = transitions["to_state"].map(STATE_LABELS)
transitions.to_csv(OUT / "state_transitions_long.csv", index=False, encoding="utf-8-sig")

audit = {
    "baseline_2011_cohort_n": int(len(wide)),
    "state_observed_by_wave": {
        str(year): int(long.loc[long["year"].eq(year), "state"].notna().sum())
        for year in [2011, 2013, 2015, 2018, 2020]
    },
    "mortality_intervals_n": int(len(mortality)),
    "mortality_unique_people_n": int(mortality["ID"].nunique()),
    "mortality_events_n": int(mortality["death"].sum()),
    "incident_adl_intervals_n": int(len(adl_incident)),
    "incident_adl_unique_people_n": int(adl_incident["ID"].nunique()),
    "incident_adl_events_n": int(adl_incident["incident_adl"].sum()),
    "transition_pairs_n": int(len(transitions)),
}
(OUT / "data_build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))

