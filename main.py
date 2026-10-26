# Reproduces the article from the survey data to its tables and figures: python main.py
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONPATH=str(ROOT / "Python"))


def run(script, *options):
    if subprocess.run([sys.executable, str(ROOT / "Python" / script), *options], cwd=ROOT, env=env).returncode:
        raise SystemExit(f"{script} failed")


run("Data_preparation/samples.py")              # results/data/
run("Data_preparation/covariates.py")
run("Data_preparation/build_model_data.py")
run("Data_preparation/build_model_data.py", "--keep137")
run("All_Models/All_Models.py")                 # results/cv/, results/fits/
run("Results/grid_predictions.py")              # results/maps/
run("Results/area_intervals.py")                # results/areas/
run("Robustness/Robustness.py")                 # results/fits/sensitivity/, results/maps/sensitivity/
run("Results/Results.py")                       # results/checks/, results/tables/, figures/
