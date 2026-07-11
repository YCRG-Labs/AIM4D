"""Shared helpers for running full-pipeline refits concurrently.

Each refit runs inside its own detached git worktree with data/ symlinked
from the canonical checkout, so concurrent folds/episodes cannot clobber
each other's stage outputs or the canonical checkout's. BLAS/ensemble
thread caps are exported into every stage subprocess; workers are plain
threads driving subprocesses, and results are collected in submission
order by the callers.

Worktrees materialize HEAD, so uncommitted changes to pipeline code are
invisible to refits; warn_if_dirty() surfaces that before a run.
"""

import os
import re
import shutil
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ["stage1_factors/extract.py", "stage2_betas/estimate.py",
          "stage3_msvar/estimate.py", "stage4_nscm/estimate.py",
          "stage5_ews/estimate.py"]

_git_lock = threading.Lock()


def default_workers():
    if os.environ.get("AIM4D_PAR"):
        return max(1, int(os.environ["AIM4D_PAR"]))
    cpus = os.cpu_count() or 4
    try:
        if sys.platform == "darwin":
            ram_gb = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True)) // 2**30
        else:
            ram_gb = (os.sysconf("SC_PAGE_SIZE")
                      * os.sysconf("SC_PHYS_PAGES")) // 2**30
    except Exception:
        ram_gb = 16
    return max(1, min(cpus // 4, ram_gb // 8))


def warn_if_dirty():
    out = subprocess.check_output(["git", "status", "--porcelain"],
                                  cwd=REPO, text=True)
    dirty = [line for line in out.splitlines()
             if line[:2].strip() and line.endswith(".py")]
    if dirty:
        print("WARNING: uncommitted .py changes are invisible to refit worktrees:")
        for line in dirty:
            print(f"  {line}")
        print()


def add_worktree(tag):
    safe = re.sub(r"\W+", "_", tag)
    path = os.path.join(REPO, ".refit_worktrees", safe)
    with _git_lock:
        if os.path.exists(path):
            subprocess.call(["git", "worktree", "remove", "--force", path],
                            cwd=REPO, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        subprocess.check_call(
            ["git", "worktree", "add", "--detach", path, "HEAD"],
            cwd=REPO, stdout=subprocess.DEVNULL)
    data_dir = os.path.join(path, "data")
    if os.path.isdir(data_dir) and not os.path.islink(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        os.symlink(os.path.join(REPO, "data"), data_dir)
    return path


def remove_worktree(path):
    with _git_lock:
        subprocess.call(["git", "worktree", "remove", "--force", path],
                        cwd=REPO, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)


def refit_env(**extra):
    env = os.environ.copy()
    threads = str(env.get("AIM4D_THREADS", "4"))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = threads
    env["AIM4D_THREADS"] = threads
    for key, value in extra.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def run_stages(worktree, env, label=""):
    for stage in STAGES:
        rc = subprocess.call([sys.executable, os.path.join(worktree, stage)],
                             env=env, cwd=worktree)
        if rc != 0:
            print(f"  [FAIL]{label} {stage} returned {rc}", flush=True)
            return rc
    return 0
