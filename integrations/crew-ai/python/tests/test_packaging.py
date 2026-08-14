"""Packaging guard: published distribution metadata must never reference a module
that the build strips.

The regression this locks down: ``[project.scripts] dev = "ag_ui_crewai.dojo:main"``
coexisted with a ``[tool.hatch.build] exclude`` that removes
``ag_ui_crewai/dojo.py`` and ``ag_ui_crewai/examples/**``, so every published wheel
and sdist installed a ``dev`` command that raised
``ModuleNotFoundError: No module named 'ag_ui_crewai.dojo'``. Nothing caught it:
the whole test suite runs against an editable install, which makes the excluded
module importable from the source tree, so the failure was invisible until a
consumer invoked the command.

crew-ai is the integration exposed to this failure mode because its dojo lives
inside the publishable package. An audit of every ``pyproject.toml`` under
``integrations/``, ``middlewares/`` and ``sdks/python`` found it was the only
library declaring a ``dev`` console script; the other twelve integrations declare
theirs inside ``python/examples/`` projects that are never published.

The check is structural rather than a build: it parses ``pyproject.toml`` and
resolves every declared entry point against the build's file-selection options,
so it costs milliseconds and belongs in the normal unit suite. It is driven
entirely by what the file says, so a new console script pointing at shipped code
passes without touching this test, and one pointing at excluded code fails.
"""

import fnmatch
import tomllib
from pathlib import Path, PurePosixPath

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"

# Hatchling options that whitelist files: whatever a target declares, the entry
# point's module has to survive it. ``packages`` and ``only-include`` are sugar
# over ``include``, and all three narrow the file set the same way.
WHITELIST_OPTIONS = ("include", "only-include", "packages")


def _load_pyproject(path=PYPROJECT):
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _matches(pattern, relpath):
    """Does hatchling's glob ``pattern`` select ``relpath``?

    Hatchling matches with gitignore syntax (via ``pathspec``), where ``*`` stops
    at a path separator. ``fnmatch`` lets ``*`` cross separators, so the two agree
    exactly on the pattern shapes actually in use here (a literal path, and a
    ``dir/**`` recursive subtree) and diverge only by fnmatch being BROADER. That
    direction is the safe one for a guard: the worst case is over-reporting, which
    surfaces as a loud test failure a maintainer inspects, never as a silently
    broken artifact. ``PurePath.match`` is not usable at all here -- before 3.13 it
    treats ``**`` as a plain ``*``, so ``ag_ui_crewai/examples/**`` would fail to
    match anything nested.

    The second arm implements the gitignore rule that a pattern naming a directory
    also selects everything beneath it, which is how ``include = ["ag_ui_crewai"]``
    ships the whole package.
    """
    pattern = pattern.strip("/")
    return fnmatch.fnmatchcase(relpath, pattern) or fnmatch.fnmatchcase(
        relpath, f"{pattern}/*"
    )


def _declared_entry_points(config):
    """Yield ``(source, name, value)`` for every entry point in ``[project]``.

    Covers ``scripts``, ``gui-scripts`` and every ``entry-points.*`` group, since
    all three land in the built distribution's ``entry_points.txt``.
    """
    project = config.get("project", {})
    for table in ("scripts", "gui-scripts"):
        for name, value in project.get(table, {}).items():
            yield f"project.{table}.{name}", name, value
    for group, entries in project.get("entry-points", {}).items():
        for name, value in entries.items():
            yield f"project.entry-points.{group}.{name}", name, value


def _build_targets(config):
    """Yield ``(source, options)`` for each hatchling build target.

    Target tables override the shared ``[tool.hatch.build]`` options per key, so
    each target is checked against its own effective file selection.
    """
    build = config.get("tool", {}).get("hatch", {}).get("build", {})
    shared = {key: value for key, value in build.items() if key != "targets"}
    targets = build.get("targets", {})
    if not targets:
        yield "tool.hatch.build", shared
        return
    for name, options in targets.items():
        yield f"tool.hatch.build.targets.{name}", {**shared, **options}


def _module_candidates(module):
    """The file paths a dotted module name can resolve to, module before package."""
    parts = module.split(".")
    return (
        PurePosixPath(*parts).with_suffix(".py"),
        PurePosixPath(*parts, "__init__.py"),
    )


def _entry_point_violations(config, package_root):
    """Every way ``config``'s entry points fail to survive its own build config.

    Returns a list of human-readable violations, empty when the invariant holds.
    """
    violations = []
    for source, _name, value in _declared_entry_points(config):
        module = value.split(":", 1)[0].strip()
        candidates = _module_candidates(module)
        resolved = next(
            (c for c in candidates if (package_root / c).is_file()),
            None,
        )
        if resolved is None:
            tried = ", ".join(str(c) for c in candidates)
            violations.append(
                f"{source} = {value!r} names module {module!r}, which has no file "
                f"in the source tree (tried {tried})"
            )
            continue

        relpath = resolved.as_posix()
        for target, options in _build_targets(config):
            for pattern in options.get("exclude", []):
                if _matches(pattern, relpath):
                    violations.append(
                        f"{source} = {value!r} resolves to {relpath}, which "
                        f"{target} strips via exclude pattern {pattern!r}"
                    )
            for option in WHITELIST_OPTIONS:
                patterns = options.get(option)
                if patterns is not None and not any(
                    _matches(p, relpath) for p in patterns
                ):
                    violations.append(
                        f"{source} = {value!r} resolves to {relpath}, which "
                        f"{target} omits: its {option} list {patterns!r} selects "
                        "nothing on that path"
                    )
    return violations


# -- the invariant, against the real pyproject.toml -------------------------


def test_declared_entry_points_survive_the_build():
    """No shipped entry point may point at a module the build strips or omits."""
    violations = _entry_point_violations(_load_pyproject(), PACKAGE_ROOT)

    assert violations == [], "\n".join(
        [
            "pyproject.toml declares entry points that the build does not ship, so "
            "the installed command would raise ModuleNotFoundError:",
            *(f"  - {v}" for v in violations),
            "Either drop the entry point (dev-only code runs via "
            "`uv run python -m <module>` from a checkout) or stop excluding the "
            "module.",
        ]
    )


# -- the checker itself, so the assertion above cannot pass vacuously ------
#
# Synthetic configs and a temporary tree: these must keep failing the way they do
# regardless of what the real pyproject.toml comes to declare.


@pytest.fixture
def fake_package(tmp_path):
    (tmp_path / "pkg" / "examples").mkdir(parents=True)
    for relpath in (
        "pkg/__init__.py",
        "pkg/shipped.py",
        "pkg/dev_only.py",
        "pkg/examples/__init__.py",
        "pkg/examples/demo.py",
    ):
        (tmp_path / relpath).write_text("")
    return tmp_path


def _config(scripts, **build_options):
    return {
        "project": {"scripts": scripts},
        "tool": {"hatch": {"build": build_options}},
    }


def test_checker_accepts_a_script_pointing_at_shipped_code(fake_package):
    config = _config(
        {"serve": "pkg.shipped:main"},
        exclude=["pkg/dev_only.py", "pkg/examples/**"],
    )

    assert _entry_point_violations(config, fake_package) == []


def test_checker_flags_a_script_excluded_by_a_literal_pattern(fake_package):
    config = _config(
        {"dev": "pkg.dev_only:main"},
        exclude=["pkg/dev_only.py", "pkg/examples/**"],
    )

    violations = _entry_point_violations(config, fake_package)

    assert len(violations) == 1
    assert "pkg/dev_only.py" in violations[0]
    assert "exclude pattern 'pkg/dev_only.py'" in violations[0]


def test_checker_flags_a_script_excluded_by_a_recursive_glob(fake_package):
    """``dir/**`` is the shape hatchling uses for demo trees, and it has to match
    nested modules, not just direct children."""
    config = _config({"demo": "pkg.examples.demo:main"}, exclude=["pkg/examples/**"])

    violations = _entry_point_violations(config, fake_package)

    assert len(violations) == 1
    assert "pkg/examples/demo.py" in violations[0]


def test_checker_flags_a_package_entry_point_excluded_as_a_directory(fake_package):
    """A dotted name can resolve to ``__init__.py``; excluding the directory still
    strips it."""
    config = _config({"demo": "pkg.examples:main"}, exclude=["pkg/examples/**"])

    violations = _entry_point_violations(config, fake_package)

    assert len(violations) == 1
    assert "pkg/examples/__init__.py" in violations[0]


def test_checker_covers_gui_scripts_and_entry_point_groups(fake_package):
    config = {
        "project": {
            "gui-scripts": {"dev-gui": "pkg.dev_only:gui"},
            "entry-points": {"some.plugin.group": {"hook": "pkg.dev_only:hook"}},
        },
        "tool": {"hatch": {"build": {"exclude": ["pkg/dev_only.py"]}}},
    }

    sources = [v.split(" = ")[0] for v in _entry_point_violations(config, fake_package)]

    assert sources == [
        "project.gui-scripts.dev-gui",
        "project.entry-points.some.plugin.group.hook",
    ]


def test_checker_flags_a_script_whose_module_does_not_exist(fake_package):
    config = _config({"ghost": "pkg.missing:main"})

    violations = _entry_point_violations(config, fake_package)

    assert len(violations) == 1
    assert "no file in the source tree" in violations[0]


def test_checker_flags_a_script_left_out_of_a_target_whitelist(fake_package):
    """An sdist ``include`` list is the other way to drop a module: the exclude
    list stays innocent and the file simply never gets selected."""
    config = {
        "project": {"scripts": {"serve": "pkg.shipped:main"}},
        "tool": {
            "hatch": {
                "build": {
                    "targets": {
                        "wheel": {"packages": ["pkg"]},
                        "sdist": {"include": ["README.md"]},
                    }
                }
            }
        },
    }

    violations = _entry_point_violations(config, fake_package)

    assert len(violations) == 1
    assert "targets.sdist omits" in violations[0]


def test_checker_applies_shared_excludes_to_every_target(fake_package):
    """``[tool.hatch.build] exclude`` with no per-target override has to be checked
    once per target, which is how the real config strips the dojo from both."""
    config = {
        "project": {"scripts": {"dev": "pkg.dev_only:main"}},
        "tool": {
            "hatch": {
                "build": {
                    "exclude": ["pkg/dev_only.py"],
                    "targets": {"wheel": {}, "sdist": {}},
                }
            }
        },
    }

    targets = [
        v.split("which ")[1].split(" strips")[0]
        for v in _entry_point_violations(config, fake_package)
    ]

    assert sorted(targets) == [
        "tool.hatch.build.targets.sdist",
        "tool.hatch.build.targets.wheel",
    ]
