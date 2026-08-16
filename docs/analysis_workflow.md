# Analysis workflow

| Order | Script | Purpose |
|---:|---|---|
| 1 | `01_build_comorbidity_state_data.py` | Harmonize repeated digestive disease, depressive symptoms, covariates, and mortality indicators. |
| 2 | `02_build_functional_multistate_data.py` | Construct living functional states and adjacent-wave interval destinations. |
| 3 | `03_run_functional_multistate_analysis.py` | Estimate observation weights and initial transition models. |
| 4 | `04_complete_multistate_analyses.py` | Produce primary relative and absolute estimates, RERI, and robustness models. |
| 5 | `05_multiple_imputation_analysis.py` | Perform iterative multiple imputation and pooled sensitivity models. |
| 6 | `06_subgroup_analysis.py` | Estimate prespecified sex, age, and hukou subgroup contrasts. |
| 7 | `07_sleep_adjustment_and_absolute_risks.py` | Add time-varying sleep adjustment and compile absolute risks. |
| 8 | `08_chronic_disease_benchmark_analysis.py` | Compare digestive disease with hypertension and diabetes benchmarks. |
| 9 | `09_temporally_ordered_sleep_pathway.py` | Evaluate the 2013 exposure → 2015 sleep → 2015–2018 transition sequence. |
| 10 | `10_strict_function_and_incident_decline.py` | Apply stricter functional definitions and incident-decline restrictions. |
| 11 | `11_build_grip_strength_dataset.py` | Construct the grip-strength inputs required by the pathway analysis. |
| 12 | `12_sequence_and_grip_pathway.py` | Assess co-occurrence sequence feasibility and the ordered grip-strength pathway. |
| 13 | `13_continuous_time_multistate_sensitivity.py` | Fit unadjusted continuous-time Markov sensitivity models with bootstrap intervals. |

Participant-level intermediate files remain local in `outputs/` and are excluded by `.gitignore`.

