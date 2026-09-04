"""Run the reproducibility pipeline in dependency order."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
PIPELINE = [
    "01_build_comorbidity_state_data.py",
    "02_build_functional_multistate_data.py",
    "03_run_functional_multistate_analysis.py",
    "04_complete_multistate_analyses.py",
    "05_multiple_imputation_analysis.py",
    "06_subgroup_analysis.py",
    "07_sleep_adjustment_and_absolute_risks.py",
    "08_chronic_disease_benchmark_analysis.py",
    "09_temporally_ordered_sleep_pathway.py",
    "10_strict_function_and_incident_decline.py",
    "11_build_grip_strength_dataset.py",
    "12_sequence_and_grip_pathway.py",
    "13_continuous_time_multistate_sensitivity.py",
    "14_dynamic_cooccurrence_history.py",
    "15_marginal_structural_sensitivity.py",
]


def main() -> None:
    data_dir = os.environ.get("CHARLS_DATA_DIR")
    if not data_dir:
        raise SystemExit("Set CHARLS_DATA_DIR before running the pipeline.")
    if not Path(data_dir).expanduser().is_dir():
        raise SystemExit("CHARLS_DATA_DIR does not point to an existing directory.")

    for name in PIPELINE:
        print(f"\n>>> Running {name}", flush=True)
        subprocess.run([sys.executable, str(CODE / name)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
