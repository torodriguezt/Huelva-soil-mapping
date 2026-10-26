# Residual checks, figures and tables.
import subprocess
import sys

from blmc.config import ROOT

HERE = ROOT / "Python" / "Results"
for script in ["residual_checks.py", "figures.py", "tables.py"]:
    if subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT).returncode:
        raise SystemExit(f"{script} failed")
