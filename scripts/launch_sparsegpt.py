"""Start the patch using the original OBC interpreter recorded on this server."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obc-root", default="outputs-obc/v4-block256")
    p.add_argument("--output-root", default="outputs-sparsegpt/v4")
    p.add_argument("--phase", choices=("all", "profile", "run"), default="all")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    ledger_path = Path(args.obc_root) / "study.json"
    if not ledger_path.exists():
        raise SystemExit(f"Completed OBC ledger missing: {ledger_path}. Use --obc-root for its actual location.")
    ledger = json.loads(ledger_path.read_text())
    attempts = [a for a in ledger["attempts"] if a["stage"] in ("run", "resume")]
    if not attempts:
        raise SystemExit("No original OBC interpreter recorded")
    command = attempts[-1]["command"]
    if command[1:3] != ["-m", "pretrained_isolation.obc.cli"]:
        raise SystemExit("Recorded OBC command is not the expected model runner")
    executable = os.environ.get("PYTHON", command[0])
    import shutil
    resolved = shutil.which(executable)
    if resolved is None:
        raise SystemExit(f"Original Python unavailable: {executable}. Set PYTHON to the original environment's Python executable.")
    env = dict(os.environ, PYTHONPATH=str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONUNBUFFERED="1")
    print(f"[LAUNCH] Using original study interpreter: {resolved}", flush=True)
    os.execve(resolved, [resolved, "-u", "-m", "pretrained_isolation.sparsegpt.study", *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
