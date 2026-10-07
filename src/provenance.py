"""A fingerprint for every result row, so a run can be resumed and two code versions never mixed.

A run is fully determined by its seed and reproduces exactly, but only while the code, the machine
and the run parameters stay the same. A row that records none of these cannot be checked, so each
row carries a fingerprint of them. That is what lets a set of runs stop halfway and resume, and
what lets a merge of two result files refuse rows produced by different code.

The fingerprint rests on a hash of the file contents, not on version control state, so an
uncommitted edit changes it exactly as a commit does. The git fields are recorded for people to
read and are not part of the fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from src.config import COST_MODEL_DIR, PROJECT_ROOT

# every file whose content can change a reported number. src/ is the library, and final_sweep.py
# holds the policy configurations and the experiment wiring, so it affects results too
_CODE_GLOBS = ("src/**/*.py", "scripts/final_sweep.py")
# the same files, in the form git status takes
_CODE_GLOB_ROOTS = ("src", "scripts/final_sweep.py")
# the scan cost curves are inputs of the Quake and bandit policies, so changing them must change
# the fingerprint of their rows
_MODEL_GLOBS = ("cost_model.yaml", "profiled_lambda_dim*.yaml")
# exact Lloyd k-means gives slightly different results at different thread counts, so the thread
# settings are part of the fingerprint
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

# run parameters that are not already columns of the row. the policy, seed and ratio identify the
# run, and everything here decides whether two runs are comparable
PARAM_KEYS = (
    "dataset", "n_ops", "initial_fraction", "query_ratio", "hot_fraction", "hot_prob",
    "delete_csf", "dedrift_n_largest",
    "workload", "cap", "target_avg", "bootstrap", "build_ratio", "window", "measure_ops",
    "max_t", "width", "eval_queries",
)

# which files can affect which rows. a change to a policy module can only affect the rows of that
# policy, so it should not make the rows of every other policy stale. everything else under src is shared,
# lire_lite.py included, because the starting index of every policy is grown under LIRE
_PRIVATE_MODULES: dict[str, tuple[str, ...]] = {
    "no_op": ("src/maintainers/no_op.py",),
    "global_rebuild": ("src/maintainers/global_rebuild.py",),
    "dedrift": ("src/maintainers/dedrift.py",),
    "cost_driven_quake": ("src/maintainers/cost_driven_quake.py",),
    "bandit": ("src/maintainers/bandit.py", "src/bandit/*.py"),
}
# the policies that price actions with the scan cost curves, the only rows the curves can affect
_LAMBDA_READERS = ("cost_driven_quake", "cost_driven_quake_tau50", "bandit")


def _family(maintainer: str) -> str | None:
    """Return the module group a policy name belongs to, or None for a policy with no own module.

    LIRE has no own module here because its file is shared, see _PRIVATE_MODULES.
    """
    if maintainer.startswith("dedrift_"):
        return "dedrift"
    if maintainer in ("bandit", "bandit_linear"):
        return "bandit"
    if maintainer == "cost_driven_quake_tau50":
        # the tau sensitivity configuration is the Quake policy with another threshold
        return "cost_driven_quake"
    return maintainer if maintainer in _PRIVATE_MODULES else None


def _code_bytes(p: Path) -> bytes:
    """Return the bytes of a file as they enter the hash.

    A Python file is hashed as its syntax tree with docstrings removed, so editing a comment or a
    docstring never changes a fingerprint. Any other file, or a Python file that does not parse,
    is hashed as raw bytes with Windows line endings normalised, so the same file hashes the same
    on Windows and Linux.
    """
    if p.suffix != ".py":
        return p.read_bytes().replace(b"\r\n", b"\n")
    import ast
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return p.read_bytes().replace(b"\r\n", b"\n")
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=False, include_attributes=False).encode()


def _hash_files(paths: list[Path]) -> str:
    """Return one hash over the relative paths and contents of `paths`, in sorted order."""
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(paths):
        h.update(str(p.relative_to(PROJECT_ROOT)).replace("\\", "/").encode())
        h.update(_code_bytes(p))
    return h.hexdigest()


def code_hash(maintainer: str | None = None) -> str:
    """Return the hash of the code that can affect a policy's rows.

    With no policy it covers every source file. With one, it covers the shared files plus that
    policy's own modules, and never another policy's modules.
    """
    files: list[Path] = []
    for pattern in _CODE_GLOBS:
        files.extend(PROJECT_ROOT.glob(pattern))
    if not files:
        raise RuntimeError("no source files matched, the fingerprint would be meaningless")
    if maintainer is None:
        return _hash_files(files)
    private = {q for pats in _PRIVATE_MODULES.values() for pat in pats
               for q in PROJECT_ROOT.glob(pat)}
    own = _family(maintainer)
    mine = ({q for pat in _PRIVATE_MODULES[own] for q in PROJECT_ROOT.glob(pat)}
            if own else set())
    return _hash_files([f for f in files if f not in private or f in mine])


def scoreboard_hash() -> str:
    """Return the hash of the calibrated price list, which prices every row of every policy."""
    path = COST_MODEL_DIR / "query_cost_calibration.yaml"
    return _hash_files([path]) if path.exists() else "absent"


def cost_model_hash() -> str:
    """Return the hash of the stored cost model files, or "absent" if there are none."""
    files: list[Path] = []
    for pattern in _MODEL_GLOBS:
        files.extend(COST_MODEL_DIR.glob(pattern))
    return _hash_files(files) if files else "absent"


def _git(*args: str) -> str:
    """Run a git command in the project folder and return its output, or "unknown" on failure."""
    try:
        out = subprocess.run(("git", *args), cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def machine_id() -> str:
    """Return a description of the machine and of the library versions that affect results.

    The machine name alone is not enough, since the library versions are what change the
    partitioning between machines. Faiss is included because the ground truth uses it when it is
    installed and falls back to scikit-learn otherwise, and the two can order exact ties
    differently.
    """
    import numpy
    import sklearn
    try:
        import faiss
        gt_backend = f"faiss{faiss.__version__}"
    except ImportError:
        gt_backend = "nofaiss"
    return "|".join((
        platform.node(), platform.machine(),
        f"py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"np{numpy.__version__}", f"sk{sklearn.__version__}", gt_backend,
    ))


def provenance(params: dict, maintainer: str | None = None) -> dict:
    """Return everything that decides whether two rows of a policy are comparable.

    Raises KeyError if a run parameter is missing from `params`.
    """
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise KeyError(f"provenance needs every run parameter, missing {missing}")
    reads_lambda = maintainer is None or maintainer in _LAMBDA_READERS
    return {
        "maintainer": maintainer or "all",
        "code_hash": code_hash() if maintainer is None else code_hash(maintainer),
        "cost_model_hash": cost_model_hash() if reads_lambda else "not read",
        "scoreboard_hash": scoreboard_hash(),
        "machine": machine_id(),
        "threads": ",".join(f"{v}={os.environ.get(v, 'unset')}" for v in _THREAD_VARS),
        "params": {k: params[k] for k in PARAM_KEYS},
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        # limited to the files the hash reads, so unrelated untracked files do not mark it dirty
        "git_dirty": bool(_git("status", "--porcelain", "--", *_CODE_GLOB_ROOTS)),
    }


def fingerprint(params: dict, maintainer: str | None = None) -> tuple[str, dict]:
    """Return the fingerprint of a policy's rows and the provenance it was computed from.

    The git fields are left out of the fingerprint, so an identical tree has the same fingerprint
    before and after a commit. The policy name is part of it, so two policies never share one.
    """
    p = provenance(params, maintainer)
    payload = {k: p[k] for k in ("maintainer", "code_hash", "cost_model_hash",
                                 "scoreboard_hash", "machine", "threads", "params")}
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode(), digest_size=8).hexdigest()
    return digest, p


def sidecar_path(output: Path) -> Path:
    """Return the path of the provenance file that goes with a result file."""
    return output.with_suffix(output.suffix + ".provenance.json")


def write_sidecar(output: Path, digest: str, prov: dict) -> None:
    """Add the provenance of a fingerprint to the provenance file of a result file.

    Entries are added rather than the file overwritten, so a resumed result file keeps the
    details of every fingerprint it contains.
    """
    path = sidecar_path(output)
    store: dict = {}
    if path.exists():
        try:
            store = json.loads(path.read_text())
        except (OSError, ValueError):
            store = {}
    store[digest] = prov
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, sort_keys=True))


def read_sidecar(output: Path) -> dict:
    """Return the provenance file of a result file, or an empty dictionary if there is none."""
    path = sidecar_path(output)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
