"""fingerprints for result rows, so runs can be resumed safely"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from src.config import COST_MODEL_DIR, PROJECT_ROOT

# files that can change a result
_CODE_GLOBS = ("src/**/*.py", "scripts/final_sweep.py")
_CODE_GLOB_ROOTS = ("src", "scripts/final_sweep.py")
# scan cost curves
_MODEL_GLOBS = ("cost_model.yaml", "profiled_lambda_dim*.yaml")
# thread settings
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

# run parameters not in the row
PARAM_KEYS = (
    "dataset", "n_ops", "initial_fraction", "query_ratio", "hot_fraction", "hot_prob",
    "delete_csf", "dedrift_n_largest",
    "workload", "cap", "target_avg", "bootstrap", "build_ratio", "window", "measure_ops",
    "max_t", "width", "eval_queries",
)

# which files affect which rows
_PRIVATE_MODULES: dict[str, tuple[str, ...]] = {
    "no_op": ("src/maintainers/no_op.py",),
    "global_rebuild": ("src/maintainers/global_rebuild.py",),
    "dedrift": ("src/maintainers/dedrift.py",),
    "cost_driven_quake": ("src/maintainers/cost_driven_quake.py",),
    "bandit": ("src/maintainers/bandit.py", "src/bandit/*.py"),
}
# policies that use the scan cost curves
_LAMBDA_READERS = ("cost_driven_quake", "cost_driven_quake_tau50", "bandit")


def _family(maintainer: str) -> str | None:
    """module group of a policy name"""
    if maintainer.startswith("dedrift_"):
        return "dedrift"
    if maintainer in ("bandit", "bandit_linear"):
        return "bandit"
    if maintainer == "cost_driven_quake_tau50":
        # Quake with tau = 50 ns
        return "cost_driven_quake"
    return maintainer if maintainer in _PRIVATE_MODULES else None


def _code_bytes(p: Path) -> bytes:
    """return the bytes of a file as they enter the hash"""
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
    """one hash over the given files"""
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(paths):
        h.update(str(p.relative_to(PROJECT_ROOT)).replace("\\", "/").encode())
        h.update(_code_bytes(p))
    return h.hexdigest()


def code_hash(maintainer: str | None = None) -> str:
    """return the hash of the code that can affect a policy's rows"""
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
    """hash of the price list"""
    path = COST_MODEL_DIR / "query_cost_calibration.yaml"
    return _hash_files([path]) if path.exists() else "absent"


def cost_model_hash() -> str:
    """return the hash of the stored cost model files, or "absent" if there are none"""
    files: list[Path] = []
    for pattern in _MODEL_GLOBS:
        files.extend(COST_MODEL_DIR.glob(pattern))
    return _hash_files(files) if files else "absent"


def _git(*args: str) -> str:
    """run a git command, or return 'unknown'"""
    try:
        out = subprocess.run(("git", *args), cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def machine_id() -> str:
    """machine and library versions"""
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
    """return everything that decides whether two rows of a policy are comparable"""
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
        # only the hashed files
        "git_dirty": bool(_git("status", "--porcelain", "--", *_CODE_GLOB_ROOTS)),
    }


def fingerprint(params: dict, maintainer: str | None = None) -> tuple[str, dict]:
    """fingerprint of a policy's rows"""
    p = provenance(params, maintainer)
    payload = {k: p[k] for k in ("maintainer", "code_hash", "cost_model_hash",
                                 "scoreboard_hash", "machine", "threads", "params")}
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode(), digest_size=8).hexdigest()
    return digest, p


def sidecar_path(output: Path) -> Path:
    """return the path of the provenance file that goes with a result file"""
    return output.with_suffix(output.suffix + ".provenance.json")


def write_sidecar(output: Path, digest: str, prov: dict) -> None:
    """add the provenance of a fingerprint to the provenance file of a result file"""
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
    """read the provenance file of a result file"""
    path = sidecar_path(output)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
