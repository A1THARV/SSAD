"""Every script under ``scripts/`` must at least parse and import.

Added after a syntax error in ``download_data.py`` survived two commits. Nothing
imported that module, so a full green test suite said nothing about whether it
could run at all -- the failure only surfaced when someone tried to use it.

These scripts are entry points rather than library code, so they have little
direct test coverage and are unlikely to gain much. A parse-and-import smoke
test is the cheap floor: it will not catch a logic error, but it catches the
class of breakage that makes a script unrunnable, which is the class that
actually happened.

Importing (not merely parsing) is deliberate: it also catches a bad import, a
typo in a module-level constant, and a dataclass whose defaults do not
construct -- all of which are module-level and therefore invisible to
``ast.parse``.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MODULES = sorted(p for p in SCRIPTS.glob("*.py") if not p.name.startswith("_"))


def test_there_are_scripts_to_check():
    # Guards against this file silently passing because the glob broke.
    assert MODULES, f"no scripts found under {SCRIPTS}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_script_parses(path: Path):
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        source = fh.read()
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(
            f"{path.name} does not parse: line {exc.lineno}: {exc.msg}\n"
            f"  {(exc.text or '').rstrip()}"
        )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_script_imports(path: Path):
    name = f"_smoke_{path.stem}"
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # Registered BEFORE exec_module, and that is not optional: @dataclass
        # resolves annotations through sys.modules[cls.__module__], which is
        # None for a module that was created but never registered. Omitting
        # this fails every script defining a dataclass, with an
        # "'NoneType' object has no attribute '__dict__'" that points nowhere
        # near the cause.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 -- any module-level failure is the point
        pytest.fail(f"{path.name} failed to import: {type(exc).__name__}: {exc}")
    finally:
        sys.modules.pop(name, None)
        if sys.path and sys.path[0] == str(SCRIPTS):
            sys.path.pop(0)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_script_does_not_run_work_on_import(path: Path):
    """A script's work must sit behind ``__main__``, not at module level.

    Otherwise importing it -- including by the test above -- would start
    downloading datasets or training models.
    """
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        tree = ast.parse(fh.read())
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    # Bootstrapping, not work. A standalone script has to put src/ on the path
    # before it can import the package, and configure_openmp must run before
    # torch loads its DLLs -- both are prerequisites for importing at all, so
    # deferring them to __main__ would defeat their purpose.
    allowed = {"configure_openmp", "insert", "append"}
    offenders = [
        ast.unparse(node.value.func)
        for node in top_level_calls
        if ast.unparse(node.value.func).split(".")[-1] not in allowed
    ]
    assert not offenders, (
        f"{path.name} calls {offenders} at module level; move it under "
        f"`if __name__ == '__main__':` so importing the module is side-effect free."
    )
