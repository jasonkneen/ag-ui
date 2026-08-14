"""Packaging guard: nothing a published artifact contains may reference a module
the build strips.

The regression this locks down: ``[project.scripts] dev = "ag_ui_crewai.dojo:main"``
coexisted with a ``[tool.hatch.build] exclude`` that removes
``ag_ui_crewai/dojo.py``, so every published wheel and sdist installed a ``dev``
command that raised ``ModuleNotFoundError: No module named 'ag_ui_crewai.dojo'``.
Nothing caught it: the suite runs against an editable install, which makes an
excluded module importable from the source tree, so the break was invisible until
a consumer ran the command. The same blind spot hides the second failure mode, a
shipped module importing an excluded one, so both are checked here.

crew-ai is the integration exposed to this because its dojo lives inside the
publishable package. Where another integration declares a ``dev`` script, it sits
in a ``python/examples/`` project that is never published.

The check is structural rather than a build, so it costs milliseconds and belongs
in the unit suite. File selection is resolved with ``pathspec.GitIgnoreSpec``, the
same library and spec class hatchling uses (``hatchling/builders/config.py``), so
the pattern semantics cannot drift from the real build: a slash-less pattern
matches at any depth, ``*`` stops at ``/``, a directory pattern takes its whole
subtree, and ``!`` re-includes. Build options this guard does not model make it
fail loudly instead of guessing, in either direction.
"""

import ast
import os
import re
from pathlib import Path, PurePosixPath

import pathspec
import pytest

try:  # tomllib is 3.11+, and requires-python still admits 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter
    import tomli as tomllib

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"
IMPORT_NAME = "ag_ui_crewai"

# Development-only code: the dojo server, the ``python -m ag_ui_crewai`` launcher
# that runs it, and the demo flows. No artifact may ship any of it, and every
# other module in the package must ship. Update these two lists (not the
# assertions) when that split changes on purpose.
DEV_ONLY_FILES = (f"{IMPORT_NAME}/dojo.py", f"{IMPORT_NAME}/__main__.py")
DEV_ONLY_TREES = (f"{IMPORT_NAME}/examples/",)

# Modules the published package exists to provide. Named so the equality check
# below cannot pass by comparing two empty sets.
CORE_MODULES = (
    f"{IMPORT_NAME}/__init__.py",
    f"{IMPORT_NAME}/endpoint.py",
    f"{IMPORT_NAME}/sdk.py",
    f"{IMPORT_NAME}/a2ui_tool.py",
)

# The hatchling file-selection options this guard mirrors.
MODELED_OPTIONS = frozenset({"exclude", "include", "only-include", "packages"})

# Options that cannot change which project files a published artifact contains:
# archive naming, metadata version, editable-install layout, target version list,
# reproducibility.
INERT_OPTIONS = frozenset(
    {
        "core-metadata-version",
        "dev-mode-dirs",
        "dev-mode-exact",
        "macos-max-compat",
        "reproducible",
        "strict-naming",
        "versions",
    }
)

# hatchling/builders/constants.py, plus the global excludes every target gets.
EXCLUDED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".venv",
        ".git",
        ".hg",
        ".hatch",
        ".tox",
        ".nox",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        ".pixi",
    }
)
EXCLUDED_FILES = frozenset({".DS_Store", ".git"})
GLOBAL_EXCLUDE_PATTERNS = ("*.py[cdo]", "/dist")


class UnmodeledBuildOption(AssertionError):
    """A build option is in play that this guard cannot resolve.

    Deliberately an ``AssertionError``, so it surfaces as a test failure telling
    the maintainer to extend the guard rather than as a verdict that happens to be
    wrong. Silent mis-reporting is the whole thing this file guards against.
    """


def _load_pyproject(path=PYPROJECT):
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _gitignore_lines(root):
    """The ``.gitignore`` hatchling folds into every exclude spec.

    ``ignore-vcs`` defaults to false, so hatchling loads the nearest
    ``.gitignore`` at or above the project root, stopping at the repo boundary
    (``hatchling.utils.fs.locate_file``). Skipping it here would make the guard
    narrower than the build and miss a module a stray ignore rule strips.
    """
    current = root
    while True:
        candidate = current / ".gitignore"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").splitlines()
        if (current / ".git").exists() or current == current.parent:
            return []
        current = current.parent


class TargetSelection:
    """Which files one hatchling build target ships.

    Mirrors ``hatchling.builders.config.BuilderConfig``: an exclude spec built
    from hatchling's global patterns, the local ``.gitignore`` and the declared
    ``exclude``; an include spec from ``include`` plus one ``/<dir>/`` pattern per
    ``packages`` entry; and ``only-include`` roots, which ``packages`` populates
    too. Files under an only-include root are selected "explicitly", which bypasses
    the include spec but not the exclude spec.
    """

    def __init__(self, source, options, root, plugin_name, project_name):
        self.source = source
        unmodeled = sorted(set(options) - MODELED_OPTIONS - INERT_OPTIONS)
        if unmodeled:
            named = ", ".join(repr(option) for option in unmodeled)
            raise UnmodeledBuildOption(
                f"{source} declares {named}, which this guard does not model, so any "
                "verdict about the files it ships would be a guess. Model the option "
                "in TargetSelection (see hatchling/builders/config.py) or drop it."
            )

        packages = [package.strip("/") for package in options.get("packages", [])]
        nested = [package for package in packages if "/" in package]
        if nested:
            raise UnmodeledBuildOption(
                f"{source} declares nested packages {nested!r}, which also make "
                "hatchling rewrite distribution paths via `sources`, so a module's "
                "shipped import path stops matching its path in the tree. Model "
                "`sources` in TargetSelection before using a src layout."
            )

        include = list(options.get("include", []))
        only_include = [
            path.strip("/") for path in options.get("only-include", [])
        ] or packages
        if plugin_name == "wheel" and not (include or only_include):
            only_include = _auto_detected_wheel_packages(root, project_name, source)

        self.only_include = only_include
        self._exclude_patterns = [
            *(("a hatchling default", p) for p in GLOBAL_EXCLUDE_PATTERNS),
            *((".gitignore", line) for line in _gitignore_lines(root)),
            *((f"{source} exclude", p) for p in options.get("exclude", [])),
        ]
        self._exclude_spec = pathspec.GitIgnoreSpec.from_lines(
            pattern for _origin, pattern in self._exclude_patterns
        )
        self._include_patterns = [*include, *(f"/{package}/" for package in packages)]
        self._include_spec = (
            pathspec.GitIgnoreSpec.from_lines(self._include_patterns)
            if self._include_patterns
            else None
        )

    def ships(self, relpath):
        return self.rejection(relpath) is None

    def rejection(self, relpath):
        """Why this target does not ship ``relpath``, or ``None`` when it does."""
        parts = PurePosixPath(relpath).parts
        if any(part in EXCLUDED_DIRECTORIES for part in parts[:-1]):
            return "sits in a directory hatchling always skips"
        if parts[-1] in EXCLUDED_FILES:
            return "is a file hatchling always skips"

        explicit = bool(self.only_include)
        if explicit and not any(
            relpath == root or relpath.startswith(f"{root}/")
            for root in self.only_include
        ):
            return f"falls outside its only-include roots {self.only_include!r}"

        if self._exclude_spec.match_file(relpath):
            return f"is stripped by {self._describe_exclusion(relpath)}"

        if (
            not explicit
            and self._include_spec is not None
            and not self._include_spec.match_file(relpath)
        ):
            return f"is not selected by its include patterns {self._include_patterns!r}"

        return None

    def _describe_exclusion(self, relpath):
        """Name the pattern that stripped ``relpath``, for the failure message.

        The verdict itself always comes from the combined spec above; this only
        looks for the last pattern to claim the path so the message can quote it,
        and stays vague when it cannot pin one down.
        """
        winner = None
        for origin, pattern in self._exclude_patterns:
            result = pathspec.GitIgnoreSpec.from_lines([pattern]).check_file(relpath)
            if result.include is True:
                winner = (origin, pattern)
            elif result.include is False:
                winner = None
        if winner is None:
            return "an exclude pattern"
        origin, pattern = winner
        return f"{origin} pattern {pattern!r}"


def _auto_detected_wheel_packages(root, project_name, source):
    """hatchling's wheel fallback when no file selection is declared: the directory
    named after the project."""
    guess = re.sub(r"[^\w\d.]+", "_", project_name)
    if guess and (root / guess / "__init__.py").is_file():
        return [guess]
    raise UnmodeledBuildOption(
        f"{source} declares no `packages`, `include` or `only-include`, and there is "
        f"no {guess or '<project name>'}/__init__.py, so hatchling falls back to "
        "src-layout or single-module detection, which this guard does not model. "
        "Declare `packages` on the target instead."
    )


def _target_selections(config, root):
    """One ``TargetSelection`` per hatchling build target.

    Target tables override the shared ``[tool.hatch.build]`` options per key. With
    no target table declared at all, both artifacts are still built from the shared
    options, so both are checked.
    """
    build = config.get("tool", {}).get("hatch", {}).get("build", {})
    shared = {key: value for key, value in build.items() if key != "targets"}
    project_name = config.get("project", {}).get("name", "")
    targets = build.get("targets") or {"wheel": {}, "sdist": {}}
    return [
        TargetSelection(
            f"tool.hatch.build.targets.{name}",
            {**shared, **options},
            root,
            name,
            project_name,
        )
        for name, options in targets.items()
    ]


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


def _module_candidates(module):
    """The file paths a dotted module name can resolve to, module before package."""
    parts = module.split(".")
    return (
        PurePosixPath(*parts).with_suffix(".py"),
        PurePosixPath(*parts, "__init__.py"),
    )


def _resolve_module(module, root):
    for candidate in _module_candidates(module):
        if (root / candidate).is_file():
            return candidate.as_posix()
    return None


def _python_files(root):
    """Every ``.py`` file in the source tree, as forward-slash relative paths."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRECTORIES]
        for filename in filenames:
            if filename.endswith(".py"):
                yield Path(dirpath, filename).relative_to(root).as_posix()


def _imported_source_files(relpath, root):
    """Source-tree files ``relpath`` imports, parent packages included.

    Imports that resolve outside the tree (third-party, stdlib) and ``from``
    targets that name a function rather than a submodule resolve to nothing and
    drop out.
    """
    tree = ast.parse((root / relpath).read_text(encoding="utf-8"), filename=relpath)
    package = PurePosixPath(relpath).parts[:-1]

    dotted = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dotted.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > len(package):
                continue
            base = list(package[: len(package) - node.level + 1]) if node.level else []
            prefix = base + (node.module.split(".") if node.module else [])
            if prefix:
                dotted.add(".".join(prefix))
            dotted.update(".".join([*prefix, alias.name]) for alias in node.names)

    imported = set()
    for name in dotted:
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            resolved = _resolve_module(".".join(parts[:depth]), root)
            if resolved is not None:
                imported.add(resolved)
    return imported - {relpath}


def _entry_point_violations(config, root):
    """Every way ``config``'s entry points fail to survive its own build config."""
    violations = []
    selections = _target_selections(config, root)
    for source, _name, value in _declared_entry_points(config):
        module = value.split(":", 1)[0].strip()
        resolved = _resolve_module(module, root)
        if resolved is None:
            tried = ", ".join(str(c) for c in _module_candidates(module))
            violations.append(
                f"{source} = {value!r} names module {module!r}, which has no file "
                f"in the source tree (tried {tried})"
            )
            continue
        for selection in selections:
            rejection = selection.rejection(resolved)
            if rejection is not None:
                violations.append(
                    f"{source} = {value!r} resolves to {resolved}, which "
                    f"{selection.source} does not ship: it {rejection}"
                )
    return violations


def _stripped_import_violations(config, root):
    """Every shipped module that imports a module its own target strips."""
    violations = []
    for selection in _target_selections(config, root):
        for relpath in sorted(_python_files(root)):
            if not selection.ships(relpath):
                continue
            for imported in sorted(_imported_source_files(relpath, root)):
                rejection = selection.rejection(imported)
                if rejection is not None:
                    violations.append(
                        f"{selection.source} ships {relpath}, which imports "
                        f"{imported}, but that file {rejection}"
                    )
    return violations


# -- the invariants, against the real pyproject.toml ------------------------


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


def test_no_shipped_module_imports_a_stripped_module():
    """The other half of the invariant: a shipped module's imports have to ship too.

    An editable install hides this completely, so a ``from .examples...`` added to
    a runtime module would only fail on a consumer's machine.
    """
    violations = _stripped_import_violations(_load_pyproject(), PACKAGE_ROOT)

    assert violations == [], "\n".join(
        [
            "modules the build ships import modules it strips, so importing the "
            "installed package would raise ModuleNotFoundError:",
            *(f"  - {v}" for v in violations),
        ]
    )


def test_every_target_ships_the_runtime_package_and_no_dev_only_module():
    """Pin the split in both directions.

    Equality, not a one-way check: dropping a path from ``exclude`` fails here just
    as loudly as excluding a module the package needs at runtime.
    """
    config = _load_pyproject()
    modules = {
        path
        for path in _python_files(PACKAGE_ROOT)
        if path.startswith(f"{IMPORT_NAME}/")
    }
    dev_only = {
        path
        for path in modules
        if path in DEV_ONLY_FILES or path.startswith(DEV_ONLY_TREES)
    }
    runtime = modules - dev_only

    assert set(DEV_ONLY_FILES) <= dev_only, "a dev-only module vanished from the tree"
    assert set(CORE_MODULES) <= runtime, "a core module vanished from the tree"
    assert len(dev_only) > len(DEV_ONLY_FILES), "the examples tree vanished"

    for selection in _target_selections(config, PACKAGE_ROOT):
        shipped = {path for path in modules if selection.ships(path)}

        assert shipped == runtime, "\n".join(
            [
                f"{selection.source} does not ship exactly the runtime modules:",
                *(
                    f"  - needed at runtime but stripped: {p}"
                    for p in sorted(runtime - shipped)
                ),
                *(f"  - dev-only but shipped: {p}" for p in sorted(shipped - runtime)),
                "Fix the build config, or update DEV_ONLY_FILES / DEV_ONLY_TREES if "
                "the split moved on purpose.",
            ]
        )


def test_the_dojo_launcher_stays_in_the_source_tree():
    """``python -m ag_ui_crewai`` is what render.yaml and the dojo runner invoke, so
    the launcher has to exist and reach the dojo even though no artifact ships it."""
    launcher = f"{IMPORT_NAME}/__main__.py"

    assert (PACKAGE_ROOT / launcher).is_file()
    # Either an import of the dojo module or the uvicorn import string that names it.
    assert f"{IMPORT_NAME}/dojo.py" in _imported_source_files(
        launcher, PACKAGE_ROOT
    ) or f"{IMPORT_NAME}.dojo" in (PACKAGE_ROOT / launcher).read_text(encoding="utf-8")


# -- the checker itself, so the assertions above cannot pass vacuously -----
#
# Synthetic configs over a temporary tree: these must keep failing the way they do
# regardless of what the real pyproject.toml comes to declare.


@pytest.fixture
def fake_package(tmp_path):
    """A miniature source tree: shipped modules, a dev-only one, a demo subtree."""
    files = {
        "pkg/__init__.py": "from .shipped import serve\n",
        "pkg/shipped.py": "import os\n\n\ndef serve():\n    pass\n",
        "pkg/dev_only.py": "from .examples import demo\n",
        "pkg/nested/__init__.py": "",
        "pkg/nested/mod.py": "",
        "pkg/examples/__init__.py": "",
        "pkg/examples/demo.py": "",
    }
    for relpath, text in files.items():
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def _config(scripts, target="wheel", **build_options):
    """A one-target config whose whitelist ships all of ``pkg``."""
    return {
        "project": {"name": "pkg", "scripts": scripts},
        "tool": {
            "hatch": {
                "build": {
                    "packages": ["pkg"],
                    **build_options,
                    "targets": {target: {}},
                }
            }
        },
    }


def test_checker_accepts_a_script_pointing_at_shipped_code(fake_package):
    config = _config(
        {"serve": "pkg.shipped:serve"},
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
    assert "pattern 'pkg/dev_only.py'" in violations[0]


def test_checker_flags_a_script_excluded_by_a_pattern_without_a_slash(fake_package):
    """The idiomatic form: gitignore patterns with no slash match at any depth, so
    ``dev_only.py`` strips ``pkg/dev_only.py``. Hand-rolled fnmatch logic anchors
    the pattern instead and calls this config green."""
    config = _config({"dev": "pkg.dev_only:main"}, exclude=["dev_only.py"])

    violations = _entry_point_violations(config, fake_package)

    assert len(violations) == 1
    assert "pattern 'dev_only.py'" in violations[0]


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


def test_checker_honours_a_negated_exclude_pattern(fake_package):
    """``!`` re-includes, so an exclude list can strip a directory and keep one file
    in it. Treating ``!`` as a literal pattern would report the kept file as gone."""
    config = _config(
        {"demo": "pkg.examples.demo:main"},
        exclude=["pkg/examples/**", "!pkg/examples/demo.py"],
    )

    assert _entry_point_violations(config, fake_package) == []


def test_checker_does_not_let_a_star_cross_a_directory_separator(fake_package):
    """``*`` stops at ``/`` in gitignore syntax, so ``pkg/*.py`` does not select a
    nested module. A matcher where ``*`` crosses separators calls this shipped and
    the module silently goes missing from the artifact."""
    config = {
        "project": {"name": "pkg", "scripts": {"serve": "pkg.nested.mod:main"}},
        "tool": {"hatch": {"build": {"targets": {"sdist": {"include": ["pkg/*.py"]}}}}},
    }

    violations = _entry_point_violations(config, fake_package)

    assert len(violations) == 1
    assert "is not selected by its include patterns" in violations[0]


def test_checker_treats_whitelist_options_as_alternatives(fake_package):
    """hatchling ORs the whitelists: ``packages`` and ``include`` feed one spec, and
    ``packages`` doubles as an only-include root. AND-ing them reports a shipped
    module as omitted because the ``include`` list says nothing about it."""
    config = {
        "project": {"name": "pkg", "scripts": {"serve": "pkg.shipped:serve"}},
        "tool": {
            "hatch": {
                "build": {
                    "targets": {
                        "wheel": {"packages": ["pkg"], "include": ["README.md"]},
                        "sdist": {"include": ["pkg", "README.md"]},
                    }
                }
            }
        },
    }

    assert _entry_point_violations(config, fake_package) == []


def test_checker_covers_gui_scripts_and_entry_point_groups(fake_package):
    config = {
        "project": {
            "name": "pkg",
            "gui-scripts": {"dev-gui": "pkg.dev_only:gui"},
            "entry-points": {"some.plugin.group": {"hook": "pkg.dev_only:hook"}},
        },
        "tool": {
            "hatch": {
                "build": {
                    "packages": ["pkg"],
                    "exclude": ["pkg/dev_only.py"],
                    "targets": {"wheel": {}},
                }
            }
        },
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
    """An sdist ``include`` list is the other way to drop a module: the exclude list
    stays innocent and the file simply never gets selected."""
    config = {
        "project": {"name": "pkg", "scripts": {"serve": "pkg.shipped:serve"}},
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
    assert "targets.sdist does not ship" in violations[0]


def test_checker_applies_shared_excludes_to_every_target(fake_package):
    """``[tool.hatch.build] exclude`` with no per-target override has to be checked
    once per target, which is how the real config strips the dojo from both."""
    config = {
        "project": {"name": "pkg", "scripts": {"dev": "pkg.dev_only:main"}},
        "tool": {
            "hatch": {
                "build": {
                    "packages": ["pkg"],
                    "exclude": ["pkg/dev_only.py"],
                    "targets": {"wheel": {}, "sdist": {}},
                }
            }
        },
    }

    targets = [
        v.split("which ")[1].split(" does not ship")[0]
        for v in _entry_point_violations(config, fake_package)
    ]

    assert sorted(targets) == [
        "tool.hatch.build.targets.sdist",
        "tool.hatch.build.targets.wheel",
    ]


def test_checker_flags_a_shipped_module_importing_a_stripped_one(fake_package):
    config = _config({}, exclude=["pkg/dev_only.py", "pkg/examples/**"])
    (fake_package / "pkg" / "shipped.py").write_text("from .examples import demo\n")

    violations = _stripped_import_violations(config, fake_package)

    stripped = "stripped by tool.hatch.build.targets.wheel exclude pattern"
    assert [v.split("ships ")[1] for v in violations] == [
        f"pkg/shipped.py, which imports pkg/examples/__init__.py, but that file is "
        f"{stripped} 'pkg/examples/**'",
        f"pkg/shipped.py, which imports pkg/examples/demo.py, but that file is "
        f"{stripped} 'pkg/examples/**'",
    ]


def test_checker_ignores_a_stripped_module_importing_a_stripped_one(fake_package):
    """``pkg/dev_only.py`` imports the demo subtree, and both are excluded, which is
    exactly the real dojo layout: nothing shipped references either."""
    config = _config({}, exclude=["pkg/dev_only.py", "pkg/examples/**"])

    assert _stripped_import_violations(config, fake_package) == []


def test_checker_refuses_an_unmodeled_build_option(fake_package):
    """``force-include`` beats ``exclude``, so guessing either way is wrong."""
    config = _config({"serve": "pkg.shipped:serve"}, **{"force-include": {"x": "y"}})

    with pytest.raises(UnmodeledBuildOption, match="'force-include'"):
        _entry_point_violations(config, fake_package)


def test_checker_refuses_a_src_layout_package(fake_package):
    config = {
        "project": {"name": "pkg", "scripts": {"serve": "pkg.shipped:serve"}},
        "tool": {"hatch": {"build": {"targets": {"wheel": {"packages": ["src/pkg"]}}}}},
    }

    with pytest.raises(UnmodeledBuildOption, match="sources"):
        _entry_point_violations(config, fake_package)


def test_checker_refuses_a_wheel_it_cannot_resolve(fake_package):
    """No whitelist and no directory named after the project: hatchling falls back
    to detection this guard does not model, so it must not answer."""
    config = {
        "project": {
            "name": "unrelated-name",
            "scripts": {"serve": "pkg.shipped:serve"},
        },
        "tool": {"hatch": {"build": {"targets": {"wheel": {}}}}},
    }

    with pytest.raises(UnmodeledBuildOption, match="src-layout"):
        _entry_point_violations(config, fake_package)


def test_checker_resolves_a_wheel_that_declares_no_file_selection(fake_package):
    """hatchling's own fallback when the package is named after the project."""
    config = {
        "project": {"name": "pkg", "scripts": {"dev": "pkg.dev_only:main"}},
        "tool": {
            "hatch": {"build": {"targets": {"wheel": {"exclude": ["pkg/dev_only.py"]}}}}
        },
    }

    violations = _entry_point_violations(config, fake_package)

    assert len(violations) == 1
    assert "pkg/dev_only.py" in violations[0]


def test_checker_applies_gitignore_patterns_like_the_build(fake_package):
    """``ignore-vcs`` defaults to false, so a ``.gitignore`` rule strips a module
    from the artifact just as an ``exclude`` entry would."""
    (fake_package / ".gitignore").write_text("dev_only.py\n")
    config = _config({"dev": "pkg.dev_only:main"})

    violations = _entry_point_violations(config, fake_package)

    assert len(violations) == 1
    assert ".gitignore pattern 'dev_only.py'" in violations[0]
