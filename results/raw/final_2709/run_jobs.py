"""Runs jobs.txt beside it, at most MAX_GIST GIST jobs and MAX_TOTAL jobs at once.

Each line is dataset, tag, comma separated seeds, comma separated maintainers. Every job writes
its own <tag>.csv and <tag>.log here with --resume, so a failed or interrupted job reruns only its
missing cells. GIST is capped at 3, a GIST process peaks near 7.4 GB and seven at once hit a
MemoryError on 25 Sept. GIST jobs are started first since they are the long pole.

    python results/raw/final_2709/run_jobs.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAX_TOTAL, MAX_GIST = 6, 3

jobs = [ln.split() for ln in (HERE / "jobs.txt").read_text().splitlines() if ln.strip()]
jobs.sort(key=lambda j: j[0] != "gist1m")
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
    n_gist = sum(j[0] == "gist1m" for j, _, _ in running)
    for job in list(jobs):
        if len(running) >= MAX_TOTAL:
            break
        if job[0] == "gist1m" and n_gist >= MAX_GIST:
            continue
        ds, tag, seeds, ms = job
        cmd = [sys.executable, "scripts/final_sweep.py", "--dataset", ds,
               "--seeds", *seeds.split(","), "--maintainers", *ms.split(","),
               "--no-figures", "--resume", "--output", str(HERE / f"{tag}.csv")]
        log = open(HERE / f"{tag}.log", "a")
        running.append((job, subprocess.Popen(cmd, cwd=REPO, stdout=log,
                                              stderr=subprocess.STDOUT, env=env), time.time()))
        jobs.remove(job)
        n_gist += ds == "gist1m"
        print(f"{stamp()} started {tag}", flush=True)
    time.sleep(20)
print(f"{stamp()} all jobs done", flush=True)
