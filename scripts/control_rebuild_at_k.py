"""The control run, the global rebuild at the partition count the bandit ended at on the same seed.

    python scripts/control_rebuild_at_k.py --seeds 42 1 7 13 23

The policies that keep K fixed hold about 176 partitions on SIFT, while the bandit grows to 408 to
546, so part of the bandit's lead may be the K it reaches rather than how it maintains partitions.
This runs the unmodified global rebuild on its usual trigger, with the partition count set to the
bandit's final K for each seed, so its query cost at every reading and the work a rebuild at that
K spends are both measured, through the same `final_sweep.run_one_staple` as every other run. The
bandit's final K per seed is read from the main results in results/raw/final_2709.

It is a control, not one of the compared policies. Its purpose was fixed before it ran, and its
result is reported whichever way it came out. The configuration is added from outside, by
extending the list of allowed names and wrapping `final_sweep.build`, so no other file changes.
Results go to results/raw/control_2809.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# the same thread pinning as final_sweep, set here too so it holds whatever the import order
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import final_sweep as fs  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.maintainers.global_rebuild import GlobalRebuildMaintainer  # noqa: E402

NAME = "global_rebuild_at_bandit_k"
SWEEP = RESULTS_DIR / "final_2709"
OUT = RESULTS_DIR / "control_2809"


def bandit_k(dataset: str) -> dict[int, int]:
    """Return the bandit's final partition count for each seed, from the main results."""
    prefix = "sift" if dataset == "sift1m" else "gist"
    d = pd.concat([pd.read_csv(p) for p in SWEEP.glob(f"{prefix}_s*.csv")])
    b = d[d.maintainer == "bandit"]
    return {int(s): int(k) for s, k in zip(b.seed, b.parts, strict=True)}


def main() -> None:
    """Run the control for the given seeds through final_sweep."""
    args = sys.argv[1:]
    dataset = args[args.index("--dataset") + 1] if "--dataset" in args else "sift1m"
    seeds = [int(s) for s in args[args.index("--seeds") + 1:] if s.lstrip("-").isdigit()]
    ks = bandit_k(dataset)
    missing = [s for s in seeds if s not in ks]
    if missing:
        raise SystemExit(f"no bandit row in {SWEEP} for seeds {missing}")
    orig = fs.build

    def build(name, seed, hi, lo, cost_model):
        """Build the control under its own name, and every other policy as final_sweep does."""
        if name == NAME:
            return GlobalRebuildMaintainer(n_partitions=ks[seed], seed=seed)
        return orig(name, seed, hi, lo, cost_model)

    fs.build = build
    fs.SENSITIVITY_VARIANTS.append(NAME)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{'sift' if dataset == 'sift1m' else 'gist'}_s{'_'.join(map(str, seeds))}"
    for s in seeds:
        print(f"seed {s}, rebuild at K {ks[s]}", flush=True)
    sys.argv = [sys.argv[0], "--dataset", dataset, "--seeds", *map(str, seeds),
                "--maintainers", NAME, "--no-figures", "--resume",
                "--output", str(OUT / f"{tag}.csv")]
    fs.main()


if __name__ == "__main__":
    main()
