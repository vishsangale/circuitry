"""Guard against release-version drift across the repo.

Single source of truth: ``circuitry.__version__`` (``src/circuitry/__init__.py``).
Every other hand-edited spot must agree with it, and ``pyproject.toml`` must keep
deriving its version dynamically from ``__version__`` so the built/published package
metadata can never drift (it was 1.4.2 in pyproject / 1.1.0 in the installed metadata
while ``__version__`` was 1.8.0 — three different numbers — before this guard existed).

Release steps live in ``.claude/skills/release-checklist/SKILL.md``. If you bumped
``__version__`` and one of these fails, you forgot a spot: fix the named file, do not
weaken the test.
"""

from __future__ import annotations

import re
from pathlib import Path

import circuitry

ROOT = Path(__file__).resolve().parent.parent
VERSION = circuitry.__version__


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION), (
        f"__version__={VERSION!r} is not a bare MAJOR.MINOR.PATCH version"
    )


def test_pyproject_version_is_dynamic_from_dunder():
    """pyproject must derive its version from ``circuitry.__version__``, never hard-code it."""
    text = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text, (
        'pyproject [project] must declare dynamic = ["version"] '
        '(do not re-introduce a static version = "..." line — it will drift)'
    )
    assert "[tool.setuptools.dynamic]" in text and re.search(
        r'version\s*=\s*\{\s*attr\s*=\s*"circuitry\.__version__"\s*\}', text
    ), 'pyproject must map the dynamic version to attr = "circuitry.__version__"'
    # Defensive: nothing should re-add a static `version = "x.y.z"` under [project].
    assert not re.search(r'(?m)^\s*version\s*=\s*"\d', text), (
        'a static version = "..." line crept back into pyproject; remove it (use the dynamic attr)'
    )


def test_readme_status_matches_version():
    readme = (ROOT / "README.md").read_text()
    assert f"v{VERSION}" in readme, (
        f"README.md is missing v{VERSION} — update the **Status:** line for this release"
    )


def test_changelog_has_section_for_version():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"## [{VERSION}]" in changelog, (
        f"CHANGELOG.md has no '## [{VERSION}]' section — add the release entry "
        "(move any Unreleased notes into it)"
    )
