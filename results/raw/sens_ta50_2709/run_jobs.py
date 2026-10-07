"""The declared target partition size check of Decision 33, `target_avg` 50, SIFT, all arms.

Runs jobs.txt beside it, at most MAX_TOTAL jobs at once, each with --target-avg 50 and its own
<tag>.csv and <tag>.log here with --resume. Runs beside the GIST tail of final_2709, which holds
three cores, so three here.

    python results/raw/sens_ta50_2709/run_jobs.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAX_TOTAL = 3
EXTRA = ["--target-avg", "50"]

jobs = [ln.split() for ln in (HERE / "jobs.txt").read_text().splitlines() if ln.strip()]
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
running: list[tuple[list[str], subprocess.Popen, float]] = []
t0 = time.time()


def stamp() -> str:
    return f"[{(time.time() - t0) / 3600:5.2f} h]"


while jobs or running:
    for job, proc, started in list(running):
        if proc.poll() is not None:
            running.remove((job, proc, started))
            print(f"{stamp()} {job[1]} exit {proc.returncode} after "
                  f"{(time.time() - started) / 60:.0f} min", flush=True)
    while jobs and len(running) < MAX_TOTAL:
        ds, tag, seeds, ms = jobs.pop(0)
        cmd = [sys.executable, "scripts/final_sweep.py", "--dataset", ds,
               "--seeds", *seeds.split(","), "--maintainers", *ms.split(","), *EXTRA,
               "--no-figures", "--resume", "--output", str(HERE / f"{tag}.csv")]
        log = open(HERE / f"{tag}.log", "a")
        running.append(([ds, tag, seeds, ms],
                        subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                         env=env), time.time()))
        print(f"{stamp()} started {tag}", flush=True)
    time.sleep(20)
print(f"{stamp()} all jobs done", flush=True)
