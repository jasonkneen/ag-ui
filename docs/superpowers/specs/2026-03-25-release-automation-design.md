# AG-UI Release Automation

**Date:** 2026-03-25
**Branch:** `feature/release-automation`
**Notion:** [Proposal: AG-UI Release Automation](https://www.notion.so/32e3aa381852818db699c9f0ee12ba77)

## Problem

AG-UI has no automated release pipeline. All npm and PyPI publishing is manual — developers run local scripts or trigger `workflow_dispatch` workflows by hand. There are no git tags and no GitHub Releases. Tyler Slaton has requested automation that:

1. Automatically releases to npm and PyPI when versions change
2. Creates GitHub Releases when publishing occurs

## Approach Evaluation

### Approach A: Changesets (CopilotKit's Model)

**How it works:** Developers add `.changeset/*.md` files in PRs describing their changes. On merge to `main`, a workflow consumes pending changesets, bumps `package.json` versions, updates changelogs, publishes to npm, and creates GitHub Releases. CopilotKit uses this with fixed versioning — all `@copilotkit/*` packages share a single version number.

**Pros:**
- Mature ecosystem with proven tooling (`@changesets/cli`)
- Automatic changelog generation from changeset descriptions
- Built-in support for fixed and independent versioning modes
- PR-based workflow gives visibility into what will be released

**Cons:**
- No native Python support — would require custom scripting for PyPI packages
- Requires developer workflow change (adding changeset files to PRs)
- AG-UI's independent versioning across ~30 packages is more complex than CopilotKit's fixed versioning
- Overkill for a project that already manages versions manually and wants to keep doing so

**Verdict:** Poor fit. The Python gap alone is disqualifying, and the workflow overhead is unnecessary given AG-UI's current manual version bumping.

### Approach B: Version-Diff Detection on Push to Main (Recommended)

**How it works:** On every push to `main`, workflows compare each `package.json` / `pyproject.toml` version against what's currently published on npm/PyPI. If the local version is higher than the published version, the package is built, tested, published, tagged, and included in a GitHub Release.

**Pros:**
- Zero developer workflow change — keep bumping versions manually as they do now
- Handles both npm and PyPI uniformly with the same detection pattern
- Idempotent — if a release fails and CI is fixed in a follow-up commit, the next push to `main` re-attempts automatically (version is still > published)
- Simple to understand, debug, and maintain
- Dynamic package discovery means no hardcoded lists to maintain

**Cons:**
- Relies on registry queries to detect changes (minor; npm and PyPI registries are highly reliable)
- No structured changelog generation (release notes will be commit-based)
- If someone accidentally bumps a version without intending to release, it will publish

**Verdict:** Best fit. Delivers exactly what's needed with minimal complexity and no workflow changes.

### Approach C: Git Tag-Triggered Releases

**How it works:** Pushing a tag like `@ag-ui/core@0.0.49` triggers a workflow that publishes that specific package. GitHub Release is created from the tag.

**Pros:**
- Explicit and intentional — nothing publishes without a deliberate tag push
- Clean audit trail — every release maps to exactly one tag
- Simple trigger mechanism

**Cons:**
- Doesn't match "automatically on version change" — requires a manual tagging step after version bump
- Developers must remember to tag correctly (scope, format, package name)
- For ~30 packages, manual tagging becomes tedious
- Doesn't solve the "version already bumped in code, now automate the rest" problem

**Verdict:** Too manual. Adds a step rather than removing one.

## Detailed Design (Approach B)

### Architecture Overview

Two GitHub Actions workflows, both triggered on push to `main` and gated on CI tests passing:

```
Push to main
    │
    ├─► CI test workflows run (existing unit-typescript-sdk.yml, unit-python-sdk.yml, etc.)
    │
    ├─► release-typescript.yml (needs: CI green)
    │     ├─ Discover publishable TS packages
    │     ├─ Compare versions against npm registry
    │     ├─ Build (pnpm run build via Nx)
    │     ├─ Publish changed packages to npm
    │     ├─ Create per-package git tags
    │     └─ Create/update GitHub Release
    │
    └─► release-python.yml (needs: CI green)
          ├─ Discover publishable Python packages
          ├─ Compare versions against PyPI
          ├─ For each changed: sync, test, build, publish
          ├─ Create per-package git tags
          └─ Create/update GitHub Release
```

### Trigger and Gating

Both release workflows trigger on `push` to `main` AND via `workflow_dispatch` (manual fallback).

**CI gating** is handled by making the release job depend on a test job within the same workflow, or by using `workflow_run` to trigger after CI completes. The simpler approach: each release workflow includes its own test step that runs the relevant test suite before publishing. This avoids complex cross-workflow dependencies while ensuring nothing publishes without green tests.

**Re-trigger behavior:** If CI fails on a version bump commit, the developer fixes the issue in a follow-up commit. On the next push to `main`, the version-diff check still sees local > published, so the release is attempted again automatically. No manual intervention needed. The `workflow_dispatch` trigger serves as a fallback for edge cases (e.g., npm outage caused the publish step to fail after tests passed).

### Workflow 1: TypeScript Release (`release-typescript.yml`)

**Trigger:** Push to `main` (paths: `sdks/typescript/**`, `middlewares/**`, `integrations/*/typescript/**`, `integrations/community/*/typescript/**`) + `workflow_dispatch`

**Concurrency:** Group `release-typescript`, `cancel-in-progress: false` (never cancel an in-progress publish).

**Steps:**

1. **Checkout** code at the push commit
2. **Setup** pnpm (via `pnpm/action-setup@v4`, version from `package.json` `packageManager` field), Node (from `package.json` engines), install deps
3. **Build** all packages: `pnpm run build` (Nx handles dependency ordering)
4. **Test** all TypeScript packages: `pnpm run test`
5. **Discover and compare versions:**
   - Scan pnpm workspace for packages where `private !== true`
   - Exclude `apps/*` packages (these are examples/demos, not publishable)
   - For each, run `npm view <package-name> version 2>/dev/null` to get the published version
   - Compare using semver: if local > published (or package is unpublished), mark for release
   - Output: list of `{name, version, path}` objects for packages that need publishing
6. **Publish** each changed package by `cd`-ing into the package directory and running `pnpm publish --no-git-checks --access public`
   - Uses `NPM_TOKEN` secret for authentication (env: `NODE_AUTH_TOKEN`)
   - `--no-git-checks` because the version is already committed
7. **Create git tags** for each published package: `<package-name>@<version>` (e.g., `@ag-ui/core@0.0.49`)
8. **Create GitHub Release** (see Release Strategy below)
9. **Push tags** to origin

**Path filtering rationale:** Only trigger when TypeScript-relevant files change. Includes `integrations/community/*/typescript/**` for community SDK packages like `@ag-ui/spring-ai`. The `workflow_dispatch` trigger bypasses path filtering for manual retries.

**Exclusions:** `apps/*` workspace packages are excluded from publishing even though they are in the pnpm workspace — they are example applications and demos, not library packages.

### Workflow 2: Python Release (`release-python.yml`)

**Trigger:** Push to `main` (paths: `sdks/python/**`, `integrations/*/python/**`, `integrations/community/*/python/**`) + `workflow_dispatch`

**Concurrency:** Group `release-python`, `cancel-in-progress: false` (never cancel an in-progress publish).

**Python package registry** — single-sourced in `scripts/release/python-packages.json`:

```json
[
  {"dir": "sdks/python", "build_system": "uv"},
  {"dir": "integrations/langgraph/python", "build_system": "uv"},
  {"dir": "integrations/crew-ai/python", "build_system": "poetry"},
  {"dir": "integrations/agent-spec/python", "build_system": "uv"},
  {"dir": "integrations/adk-middleware/python", "build_system": "uv"},
  {"dir": "integrations/aws-strands/python", "build_system": "uv"},
  {"dir": "integrations/claude-agent-sdk/python", "build_system": "uv"},
  {"dir": "integrations/langroid/python", "build_system": "uv"}
]
```

This registry is the single source of truth for which Python packages are publishable. New packages are added here. The `build_system` field controls how each package is built and published (see Build System Variants below).

**Steps:**

1. **Checkout** code
2. **Setup** `uv` (>=0.8.0), Python, and Poetry (for Poetry-based packages)
3. **Discover and compare versions:**
   - Read package list from `scripts/release/python-packages.json`
   - For each directory, extract name and version from `pyproject.toml`:
     - For `uv` packages: read `[project].name` and `[project].version`
     - For `poetry` packages: read `[tool.poetry].name` and `[tool.poetry].version`
   - Query PyPI JSON API (`https://pypi.org/pypi/<name>/json`) to get published version
   - Compare: if local > published (or package is unpublished), mark for release
4. **For each changed package (uv build system):**
   - `uv sync` (install dependencies)
   - Run tests (extract test command from `[tool.ag-ui.scripts].test` in `pyproject.toml`, matching existing pattern)
   - `uv build` (create wheel and sdist)
   - Verify wheel permissions (matching existing `publish-python-package.yml` pattern)
   - `uv publish` (env: `UV_PUBLISH_TOKEN: ${{ secrets.PYPI_API_TOKEN }}`)
5. **For each changed package (poetry build system):**
   - `poetry install` (install dependencies)
   - Run tests if `[tool.ag-ui.scripts].test` exists (see Test Configuration below)
   - `poetry build` (create wheel and sdist)
   - Verify wheel permissions
   - `uv publish` on the built artifacts (env: `UV_PUBLISH_TOKEN: ${{ secrets.PYPI_API_TOKEN }}`) — uv can publish any wheel regardless of build system
6. **Create git tags** for each published package: `<package-name>@<version>` (e.g., `ag-ui-protocol@0.1.15`)
7. **Create GitHub Release** (see Release Strategy below)
8. **Push tags** to origin

**Build system variants:** The `crew-ai` integration uses Poetry (`poetry.core.masonry.api`) as its build backend instead of `uv`. The workflow detects this from the `build_system` field in `python-packages.json` and uses the appropriate build toolchain. All other Python packages use `uv`. Publishing is always done via `uv publish` regardless of build system, since it can publish any standard wheel.

**Test configuration:** Tests are run only if `[tool.ag-ui.scripts].test` exists in the package's `pyproject.toml`. If absent, the test step is skipped with a warning in the workflow logs. This is intentional — some packages (e.g., `crew-ai`, `claude-agent-sdk`) currently have no tests. The existing `publish-python-package.yml` hard-fails on missing test config, but the automated workflow takes a more permissive approach: a package without tests can still be published, but the missing test config is logged as a warning. This avoids blocking releases for packages that are already being published manually without tests today.

**Prerequisite:** The following packages currently lack `[tool.ag-ui.scripts].test` and will publish without test validation until test configs are added:
- `integrations/crew-ai/python` (no tests exist)
- `integrations/claude-agent-sdk/python` (no test config)
- `integrations/aws-strands/python` (no test config)

Adding test configs to these packages is recommended but not a blocker for this automation.

**Python package discovery:** Unlike TypeScript where the pnpm workspace defines publishable packages, Python packages are registered in `scripts/release/python-packages.json`. New Python packages require adding an entry to this file — this is intentional to prevent accidental publishing of work-in-progress packages.

### GitHub Release Strategy

**Tags:** Per-package git tags for traceability (e.g., `@ag-ui/core@0.0.49`, `ag-ui-protocol@0.1.15`).

**Releases:** One consolidated GitHub Release per workflow run that publishes at least one package. Both the TypeScript and Python workflows can create releases independently.

**Release tag format:** `release/YYYY-MM-DD` — one release per day. Multiple pushes on the same day append to the same release. The release body is cumulative: each workflow run adds a timestamped section listing the packages it published, so the full release shows everything published that day.

**Release body format:**
```markdown
## Packages Published

### TypeScript (npm)
| Package | Version | Install |
|---------|---------|---------|
| @ag-ui/core | 0.0.49 | `npm install @ag-ui/core@0.0.49` |
| @ag-ui/client | 0.0.49 | `npm install @ag-ui/client@0.0.49` |

### Python (PyPI)
| Package | Version | Install |
|---------|---------|---------|
| ag-ui-protocol | 0.1.15 | `pip install ag-ui-protocol==0.1.15` |

## Changes
<git log --oneline since last release tag>
```

**Implementation:** Use the `gh` CLI for release management, with explicit create-or-update logic:

```bash
# Try to get existing release for today's tag
TAG="release/$(date +%Y-%m-%d)"
if gh release view "$TAG" &>/dev/null; then
  # Release exists — append our section to the body
  EXISTING_BODY=$(gh release view "$TAG" --json body -q .body)
  gh release edit "$TAG" --notes "${EXISTING_BODY}${NEW_SECTION}"
else
  # Create new release
  gh release create "$TAG" --title "Release $(date +%Y-%m-%d)" --notes "$BODY"
fi
```

This handles the race condition between TypeScript and Python workflows: whichever completes first creates the release, the second appends to it. The `gh release view` check is lightweight and idempotent. If both workflows race on creation, one will get a conflict error — the workflow retries with the update path.

### Secrets Required

| Secret | Purpose | Status |
|--------|---------|--------|
| `NPM_TOKEN` | npm publish (env: `NODE_AUTH_TOKEN`) | **Needs to be added** |
| `PYPI_API_TOKEN` | PyPI publish (env: `UV_PUBLISH_TOKEN`) | Already exists (used by `publish-python-package.yml`) |

### Permissions

Both workflows need:
- `contents: write` — to create git tags and GitHub Releases
- `id-token: write` — for potential future OIDC trusted publishing

### What Stays the Same

- **Developer workflow:** Bump versions manually in `package.json` / `pyproject.toml`, open PR, merge to `main`
- **`publish-commit.yml`:** pkg-pr-new preview packages on every push/PR — unchanged
- **`build-python-preview.yml` / `publish-python-preview.yml`:** TestPyPI preview builds on PRs — unchanged
- **`publish-python-package.yml`:** Manual Python publish with CODEOWNERS check — stays as fallback
- **Manual TS publish scripts:** `pnpm run publish` in root `package.json` — stays as fallback

### What's New

| File | Purpose |
|------|---------|
| `.github/workflows/release-typescript.yml` | Auto-publish TS packages to npm on version change |
| `.github/workflows/release-python.yml` | Auto-publish Python packages to PyPI on version change |
| `scripts/release/python-packages.json` | Single-sourced registry of publishable Python packages |
| `scripts/release/detect-ts-version-changes.sh` | Compare TS package versions against npm registry |
| `scripts/release/detect-py-version-changes.sh` | Compare Python package versions against PyPI |
| `scripts/release/create-or-update-release.sh` | Create or update daily GitHub Release |

### Edge Cases

1. **npm/PyPI outage during publish:** Publish step fails, workflow fails. Next push to `main` re-triggers because version is still > published. `workflow_dispatch` available for immediate retry.

2. **Accidental version bump:** If someone bumps a version without intending to release, it will publish on merge. Mitigation: version bumps should be intentional and reviewed in PR. This is consistent with the existing manual workflow where a version bump means "this is ready to publish."

3. **Partial publish failure:** If 3 of 5 packages publish and the 4th fails, the already-published packages are on the registry. On retry, version-diff detects only the unpublished packages (their local version is still > published). The successfully published packages are skipped. Git tags and GitHub Release are only created for successfully published packages.

4. **Race condition between TS and Python workflows:** Both may try to create a GitHub Release with the same `release/YYYY-MM-DD` tag. Handled via `gh release view` check + create-or-update logic with retry on conflict (see Release Strategy implementation above). Whichever workflow completes first creates the release; the second appends its section.

5. **New package added:** A new TypeScript package in the workspace with `private: false` is automatically discovered. A new Python package requires adding its directory to the workflow's scan list.

6. **Pre-release / alpha versions:** The version-diff check uses semver comparison. Publishing `0.0.49-alpha.1` when `0.0.48` is on npm will trigger a release. The workflow should tag alpha releases with `--tag alpha` on npm (detect from version string containing `-`).

### Testing Plan

1. **Dry-run mode:** Both workflows support a `dry_run` input via `workflow_dispatch` that runs all steps except the actual `publish` and `tag` commands. This allows validating the version detection and build steps without side effects.

2. **Initial validation:** Run both workflows in dry-run mode on the `feature/release-automation` branch via `workflow_dispatch` before merging.

3. **First real release:** After merging, bump one TS SDK package and one Python package version in a PR. Merge and observe that both workflows detect the change, publish, tag, and create a GitHub Release.
