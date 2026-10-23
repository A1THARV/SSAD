"""H — reproduce the paper's numbers from a fresh clone (INDICON revision).

    python scripts/revision_fresh_clone_test.py

Reviewer 3 asks for access to the code and configuration. The claim worth making
is not "the code is on GitHub" but "a fresh clone reproduces the numbers", and
that is only worth making if it has been tried.

What this does:

1. Clones the repository into a new directory from the local git history, so the
   clone contains exactly what a reader would receive -- no untracked files, no
   caches, no datasets.
2. Installs it into an isolated environment and runs the test suite there.
3. Runs `download_data.py --verify` in the clone with NO data, and checks it
   reports every source as missing with an actionable path rather than failing
   obscurely. This is the state a reader starts in.
4. Links the clone's `data/` at the already-downloaded datasets -- the datasets
   are not redistributable, so a genuine re-download cannot be part of an
   automated check -- and re-runs the commands behind each table.
5. Diffs the numbers the clone produces against the committed results.

Step 4 is the honest limit of this test: it verifies the CODE reproduces the
numbers given the data, not that the data can be re-fetched unattended. HAD100
needs a Google Drive confirmation token, FLAME 3 and ECOSTRESS are
account-gated. Those are reported as manual steps, which is what
`download_data.py` exists to make survivable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: (label, argv, results file, keys to compare). Each is a command the README
#: tells a reader to run, paired with the number it must produce.
CHECKS = [
    ("E benchmark table",
     [sys.executable, "scripts/revision_benchmark_table.py"],
     "results/revision/e_benchmark_table.json",
     [("ours_rank", None), ("rows", "len")]),
    ("F/H evidence tables",
     [sys.executable, "scripts/revision_tables.py"],
     None, []),
    ("A2/B statistics",
     [sys.executable, "scripts/revision_stats.py"],
     "results/revision/a2b_stats.json",
     [("scene_level_mauc", None), ("b_alpha_cv.alpha_selected_per_fold", None),
      ("a2_paired_tests", "len")]),
]


def run(cmd, cwd, env=None, timeout=3600):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def dig(obj, dotted):
    for part in dotted.split("."):
        obj = obj[part]
    return obj


def link_data(clone: Path, real: Path) -> str:
    """Point the clone's data/ at the real datasets without copying 140 GB."""
    target = clone / "data"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    if os.name == "nt":
        code, out = run(["cmd", "/c", "mklink", "/J", str(target), str(real)], clone)
        if code == 0:
            return "junction"
    try:
        target.symlink_to(real, target_is_directory=True)
        return "symlink"
    except OSError:
        return "FAILED"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/revision/h_fresh_clone.json")
    p.add_argument("--keep", action="store_true", help="keep the clone directory")
    args = p.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="ssad_fresh_"))
    clone = tmp / "SSAD"
    report = {"clone_dir": str(clone), "steps": []}

    def step(name, ok, detail=""):
        report["steps"].append({"step": name, "ok": bool(ok), "detail": detail[-1500:]})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail[:140]}" if detail else ""))
        return ok

    print(f"fresh clone -> {clone}\n")
    code, out = run(["git", "clone", "--quiet", str(REPO), str(clone)], tmp)
    if not step("git clone", code == 0, out):
        return 1

    tracked = [p for p in clone.rglob("*") if p.is_file() and ".git" not in p.parts]
    step("clone contains no dataset files",
         not any("data" == q.parts[0] for q in
                 (f.relative_to(clone) for f in tracked)),
         f"{len(tracked)} files, {sum(f.stat().st_size for f in tracked) / 1e6:.1f} MB")

    env = dict(os.environ, PYTHONPATH=str(clone / "src"), KMP_DUPLICATE_LIB_OK="TRUE")

    # 1 -- the state a reader starts in: no data at all.
    code, out = run([sys.executable, "scripts/download_data.py", "--verify"], clone, env)
    missing_named = all(k in out for k in ("place at", "needed by"))
    step("download_data.py --verify reports missing data actionably",
         code != 0 and missing_named,
         f"exit {code}; names paths and milestones: {missing_named}")

    # 2 -- the test suite, in the clone.
    code, out = run([sys.executable, "-m", "pytest", "-q", "--no-header"], clone, env,
                    timeout=1800)
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    step("pytest in the clone", code == 0, tail)

    # 3 -- link the real datasets and reproduce each table.
    how = link_data(clone, REPO / "data")
    if not step(f"link data/ into the clone ({how})", how != "FAILED", how):
        return 1

    for label, cmd, results_file, keys in CHECKS:
        code, out = run(cmd, clone, env, timeout=3600)
        if not step(f"run: {label}", code == 0, out.strip().splitlines()[-1] if out else ""):
            continue
        if not results_file:
            continue
        a = json.loads((clone / results_file).read_text(encoding="utf-8"))
        b_path = REPO / results_file
        if not b_path.is_file():
            step(f"compare: {label}", False, "no committed baseline to compare against")
            continue
        b = json.loads(b_path.read_text(encoding="utf-8"))
        diffs = []
        for key, mode in keys:
            try:
                va, vb = dig(a, key), dig(b, key)
                if mode == "len":
                    va, vb = len(va), len(vb)
                if json.dumps(va, sort_keys=True) != json.dumps(vb, sort_keys=True):
                    diffs.append(f"{key}: clone={va} vs repo={vb}")
            except Exception as exc:
                diffs.append(f"{key}: {exc}")
        step(f"compare: {label}", not diffs,
             "identical" if not diffs else "; ".join(diffs))

    ok = all(s["ok"] for s in report["steps"])
    report["all_passed"] = ok
    report["note"] = (
        "Datasets are linked rather than re-downloaded: they are not "
        "redistributable and two of the five are account-gated, so an unattended "
        "re-download cannot be part of an automated check. This verifies the CODE "
        "reproduces the numbers given the data, and separately that "
        "download_data.py tells a reader exactly what to fetch and where to put it.")
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'ALL PASSED' if ok else 'SOME STEPS FAILED'} -> {out_path}")
    if args.keep:
        print(f"clone kept at {clone}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
