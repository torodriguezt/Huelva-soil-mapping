# Cross-validation of the candidates and the univariate baseline, then the full-data fit of the selected model.
import subprocess
import sys

from blmc.config import DIR_CV, FULL_ITERATIONS, ROOT
from blmc.io import read_json

HERE = ROOT / "Python" / "All_Models"


def python(script, *options):
    if subprocess.run([sys.executable, str(HERE / script), *options], cwd=ROOT).returncode:
        raise SystemExit(f"{script} failed")


python("cross_validation.py")
python("univariate_baseline.py")
selected = read_json(DIR_CV / "selected_model.json")["selected"]
python("fit_model.py", "--model", selected, *[f"--{k}={v}" for k, v in FULL_ITERATIONS.items()])
