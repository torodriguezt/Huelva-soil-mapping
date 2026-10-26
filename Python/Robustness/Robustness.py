# Prior, record and element-group sensitivity refits of the selected model.
import subprocess
import sys

from blmc.config import DIR_CV, ROOT, SENSITIVITIES, SENSITIVITY_ITERATIONS
from blmc.io import read_json

PYTHON = ROOT / "Python"


def run_together(commands):
    running = [subprocess.Popen([sys.executable, *command], cwd=ROOT) for command in commands]
    failed = [command for command, process in zip(commands, running) if process.wait()]
    if failed:
        raise SystemExit(f"failed: {failed}")


selected = read_json(DIR_CV / "selected_model.json")["selected"]
lengths = [f"--{k}={v}" for k, v in SENSITIVITY_ITERATIONS.items()]
run_together([[str(PYTHON / "All_Models" / "fit_model.py"), "--model", selected, "--name", f"sensitivity/{name}", *options,
               *lengths] for name, options in SENSITIVITIES.items()])
run_together([[str(PYTHON / "Results" / "grid_predictions.py"), "--fit", f"sensitivity/{name}"] for name in SENSITIVITIES])
run_together([[str(PYTHON / "Robustness" / "sensitivity_review.py")]])
