# CHARLS digestive disease, depressive symptoms, and functional transitions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21966943.svg)](https://doi.org/10.5281/zenodo.21966943)

This repository contains the statistical analysis code for a longitudinal study of time-varying digestive disease–depressive symptom states and transitions among no functional limitation, IADL limitation, ADL limitation, and death in CHARLS.

## Privacy and data access

No participant-level data, derived participant-level datasets, manuscript files, or author contact information are included. CHARLS data cannot be redistributed from this repository. Researchers must register with the [CHARLS project](https://charls.charlsdata.com/) and obtain the applicable survey files under the CHARLS terms of use.

## Reproducibility scope

The scripts reproduce:

- construction of repeated exposure states and multistate functional intervals;
- inverse-observation-weighted modified Poisson transition models;
- standardized absolute risks and additive-interaction estimates;
- chronic-disease benchmark comparisons;
- sleep and grip-strength pathway analyses;
- subgroup, multiple-imputation, strict-outcome, incident-decline, and continuous-time sensitivity analyses.

The code expects the original CHARLS directory structure used in the public releases. Minor path adjustments may be required if downloaded files were renamed or reorganized.

## Setup

1. Install Python 3.11 or later.
2. Create an isolated environment and install the packages in `requirements.txt`.
3. Set `CHARLS_DATA_DIR` to the local directory containing the CHARLS files.
4. Run the scripts in the order shown in `run_all.py`.

PowerShell example:

```powershell
$env:CHARLS_DATA_DIR = "C:\path\to\CHARLS"
python run_all.py
```

Bash example:

```bash
export CHARLS_DATA_DIR=/path/to/CHARLS
python run_all.py
```

Outputs are written under `outputs/` and are ignored by Git. The pipeline stops with an error if `CHARLS_DATA_DIR` is not defined.

## Directory layout

```text
code/                 analysis scripts
docs/                 run order and variable mapping
outputs/              locally generated files; never committed
run_all.py             sequential pipeline runner
requirements.txt       Python dependencies
```

## Statistical notes

The main models estimate interval transition risk ratios with modified Poisson regression and participant-clustered standard errors. Stabilized inverse observation probability weights address unknown destination states. Analyses are associational and do not identify causal effects.

## Versioning and citation

Stable versions are distributed through GitHub releases and archived in Zenodo. Cite version 1.0.0 as:

> Liang H, Wu X, Huang S, Zheng F, Qiu R. Analysis code for digestive disease–depressive symptom co-occurrence and functional transitions in CHARLS. Version 1.0.0. Zenodo. 2026. https://doi.org/10.5281/zenodo.21966943

The concept DOI for all versions is https://doi.org/10.5281/zenodo.21966942.
