# Prior and record sensitivity refits of the selected model.
import subprocess
import sys

from blmc.config import DIR_CV, ROOT, SENSITIVITIES, SENSITIVITY_ITERATIONS
from blmc.io import read_json

selected = read_json(DIR_CV / "selected_model.json")["selected"]
fit_model = ROOT / "Python" / "All_Models" / "fit_model.py"
lengths = [f"--{k}={v}" for k, v in SENSITIVITY_ITERATIONS.items()]
running = {name: subprocess.Popen([sys.executable, str(fit_model), "--model", selected, "--name", f"sensitivity/{name}",
                                   *options, *lengths], cwd=ROOT)
           for name, options in SENSITIVITIES.items()}
failed = [name for name, process in running.items() if process.wait()]
if failed:
    raise SystemExit(f"sensitivity fits failed: {failed}")

review = ROOT / "Python" / "Robustness" / "sensitivity_review.py"
if subprocess.run([sys.executable, str(review)], cwd=ROOT).returncode:
    raise SystemExit("sensitivity_review.py failed")
