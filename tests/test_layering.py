"""Layering rules from docs/design.md §3. Belt-and-suspenders with the
import-linter config — the unit test catches violations at pytest time and
gives a clearer error than import-linter's CLI output."""

from __future__ import annotations

import ast
import pathlib
import sys

SRC = pathlib.Path(__file__).parent.parent / "src" / "circuitry"

FORBIDDEN = {
    "core": ("circuitry.recorder", "circuitry.recipes", "circuitry.writers", "circuitry.cli"),
    "recipes": ("circuitry.cli",),
}

# Allowlist for the reverse-dependency rule (§3): `circuitry` may only import
# from itself, the standard library, or its declared third-party deps. Any
# other root package implies an unauthorized dependency on a consumer codebase
# or an undeclared third-party — both should fail this test.
ALLOWED_ROOTS = frozenset({"circuitry", "torch", "numpy", "tensorboard"}) | sys.stdlib_module_names


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_core_does_not_import_higher_layers():
    for py in (SRC / "core").rglob("*.py"):
        for imp in _imports(py):
            for forbidden in FORBIDDEN["core"]:
                assert not imp.startswith(forbidden), (
                    f"core/{py.relative_to(SRC / 'core')} imports {imp}, "
                    f"violating §3 layering rule"
                )


def test_recipes_do_not_import_cli():
    for py in (SRC / "recipes").rglob("*.py"):
        for imp in _imports(py):
            for forbidden in FORBIDDEN["recipes"]:
                assert not imp.startswith(forbidden), (
                    f"recipes/{py.relative_to(SRC / 'recipes')} imports {imp}, "
                    f"violating §3 layering rule"
                )


def test_only_approved_root_packages_imported():
    """Reverse-dependency rule (§3): circuitry is the consumed dep, never the
    consumer. Implemented as an allowlist — any new root package is a red flag
    that the author should add to ``ALLOWED_ROOTS`` explicitly if intentional.
    """
    for py in SRC.rglob("*.py"):
        for imp in _imports(py):
            root = imp.split(".", 1)[0]
            assert root in ALLOWED_ROOTS, (
                f"{py.relative_to(SRC)} imports {imp!r} — root package {root!r} "
                f"is not in ALLOWED_ROOTS. Either it's an unauthorized "
                f"consumer-codebase import (§3 reverse-dependency rule) or a "
                f"new third-party dep that needs declaring in pyproject.toml "
                f"and added to ALLOWED_ROOTS in this test."
            )
