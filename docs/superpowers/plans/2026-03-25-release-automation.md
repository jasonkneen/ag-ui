# AG-UI Release Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically publish TypeScript packages to npm and Python packages to PyPI when version numbers are bumped on the `main` branch, with GitHub Release creation.

**Architecture:** Two GitHub Actions workflows (`release-typescript.yml`, `release-python.yml`) trigger on push to `main`, compare local package versions against published registry versions, and publish any packages where local > published. Shared shell scripts handle version detection and GitHub Release management. Both workflows gate on tests passing and support manual `workflow_dispatch` with dry-run mode.

**Tech Stack:** GitHub Actions, pnpm, Nx, uv, Poetry, npm CLI, PyPI JSON API, `gh` CLI

**Spec:** `docs/superpowers/specs/2026-03-25-release-automation-design.md`
**Notion:** https://www.notion.so/32e3aa381852818db699c9f0ee12ba77

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/release/python-packages.json` | Single-sourced registry of publishable Python packages with build system metadata |
| `scripts/release/detect-ts-version-changes.sh` | Discovers publishable TS packages from pnpm workspace, compares against npm, outputs JSON of changed packages |
| `scripts/release/detect-py-version-changes.sh` | Reads python-packages.json, extracts versions from pyproject.toml (uv or poetry), compares against PyPI, outputs JSON |
| `scripts/release/create-or-update-release.sh` | Creates or appends to a daily GitHub Release (`release/YYYY-MM-DD`) with published package info |
| `.github/workflows/release-typescript.yml` | Orchestrates TS build → test → detect → publish → tag → release on push to main |
| `.github/workflows/release-python.yml` | Orchestrates Python detect → per-package (sync → test → build → publish) → tag → release on push to main |

---

### Task 1: Create Python Package Registry

**Files:**
- Create: `scripts/release/python-packages.json`

- [ ] **Step 1: Create the registry file**

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

- [ ] **Step 2: Validate the registry against actual repo**

Run from repo root:
```bash
for dir in $(jq -r '.[].dir' scripts/release/python-packages.json); do
  if [ -f "$dir/pyproject.toml" ]; then
    echo "OK: $dir"
  else
    echo "MISSING: $dir/pyproject.toml"
  fi
done
```
Expected: All entries print "OK"

- [ ] **Step 3: Commit**

```bash
git add scripts/release/python-packages.json
git commit -m "Add Python package registry for release automation"
```

---

### Task 2: Create TypeScript Version Detection Script

**Files:**
- Create: `scripts/release/detect-ts-version-changes.sh`

This script discovers all publishable TypeScript packages from the pnpm workspace, compares each version against what's published on npm, and outputs a JSON array of packages that need publishing.

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# scripts/release/detect-ts-version-changes.sh
#
# Discovers publishable TypeScript packages from the pnpm workspace,
# compares each version against npm, and outputs a JSON array of
# packages that need publishing.
#
# Output format (stdout): [{"name":"@ag-ui/core","version":"0.0.49","path":"sdks/typescript/packages/core"}, ...]
# Logs go to stderr so they don't corrupt the JSON output.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Get all workspace packages as JSON
PACKAGES=$(cd "$REPO_ROOT" && pnpm list -r --json --depth=-1 2>/dev/null)

# Iterate over each package
echo "$PACKAGES" | jq -c '.[]' | while read -r pkg; do
  NAME=$(echo "$pkg" | jq -r '.name')
  VERSION=$(echo "$pkg" | jq -r '.version')
  PKG_PATH=$(echo "$pkg" | jq -r '.path')
  RELATIVE_PATH="${PKG_PATH#$REPO_ROOT/}"
  PRIVATE=$(echo "$pkg" | jq -r '.private // false')

  # Skip private packages
  if [ "$PRIVATE" = "true" ]; then
    echo "SKIP (private): $NAME" >&2
    continue
  fi

  # Skip apps/* packages (examples/demos, not publishable libraries)
  if [[ "$RELATIVE_PATH" == apps/* ]]; then
    echo "SKIP (app): $NAME" >&2
    continue
  fi

  # Skip mastra examples
  if [[ "$RELATIVE_PATH" == *examples* ]]; then
    echo "SKIP (example): $NAME" >&2
    continue
  fi

  # Query npm for the published version
  PUBLISHED_VERSION=$(npm view "$NAME" version 2>/dev/null || echo "")

  if [ -z "$PUBLISHED_VERSION" ]; then
    echo "NEW (unpublished): $NAME@$VERSION at $RELATIVE_PATH" >&2
    echo "{\"name\":\"$NAME\",\"version\":\"$VERSION\",\"path\":\"$RELATIVE_PATH\"}"
  else
    # Compare versions using inline Node.js (no external semver dependency)
    IS_NEWER=$(node -e "
      const local = '$VERSION'.split(/[-+]/)[0].split('.').map(Number);
      const pub = '$PUBLISHED_VERSION'.split(/[-+]/)[0].split('.').map(Number);
      for (let i = 0; i < 3; i++) {
        if ((local[i]||0) > (pub[i]||0)) { console.log('true'); process.exit(); }
        if ((local[i]||0) < (pub[i]||0)) { console.log('false'); process.exit(); }
      }
      // Same base version: only upgrade if local is stable and published is prerelease
      const localPre = '$VERSION'.includes('-');
      const pubPre = '$PUBLISHED_VERSION'.includes('-');
      if (!localPre && pubPre) { console.log('true'); } // stable replaces prerelease
      else { console.log('false'); } // same base, don't compare prerelease ordering
    " 2>/dev/null || echo "false")

    if [ "$IS_NEWER" = "true" ]; then
      echo "CHANGED: $NAME $PUBLISHED_VERSION -> $VERSION at $RELATIVE_PATH" >&2
      echo "{\"name\":\"$NAME\",\"version\":\"$VERSION\",\"path\":\"$RELATIVE_PATH\"}"
    else
      echo "UP-TO-DATE: $NAME@$VERSION (published: $PUBLISHED_VERSION)" >&2
    fi
  fi
done | jq -sc '.'
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/release/detect-ts-version-changes.sh
```

- [ ] **Step 3: Test locally**

Run from repo root (requires `pnpm install` and `npm` available):
```bash
bash scripts/release/detect-ts-version-changes.sh 2>/dev/null | jq .
```
Expected: A JSON array. If all packages are already published at their current versions, the array will be empty `[]`. Stderr will show which packages were checked and their status.

Also test stderr logging:
```bash
bash scripts/release/detect-ts-version-changes.sh >/dev/null
```
Expected: Lines like `UP-TO-DATE: @ag-ui/core@0.0.49 (published: 0.0.49)` for each package.

- [ ] **Step 4: Verify version comparison works**

The script uses inline Node.js for version comparison (no external dependencies). Verify:
```bash
node -e "
const local = '0.0.49'.split(/[-+]/)[0].split('.').map(Number);
const pub = '0.0.48'.split(/[-+]/)[0].split('.').map(Number);
for (let i = 0; i < 3; i++) {
  if ((local[i]||0) > (pub[i]||0)) { console.log('true'); process.exit(); }
  if ((local[i]||0) < (pub[i]||0)) { console.log('false'); process.exit(); }
}
console.log('false');
"
```
Expected: `true`

- [ ] **Step 5: Commit**

```bash
git add scripts/release/detect-ts-version-changes.sh
git commit -m "Add TypeScript version detection script for release automation"
```

---

### Task 3: Create Python Version Detection Script

**Files:**
- Create: `scripts/release/detect-py-version-changes.sh`

This script reads `python-packages.json`, extracts name/version from each package's `pyproject.toml` (handling both uv and poetry build systems), compares against PyPI, and outputs JSON.

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# scripts/release/detect-py-version-changes.sh
#
# Reads scripts/release/python-packages.json, extracts name and version
# from each package's pyproject.toml (uv or poetry format), compares
# against PyPI, and outputs a JSON array of packages that need publishing.
#
# Output format (stdout): [{"name":"ag-ui-protocol","version":"0.1.15","dir":"sdks/python","build_system":"uv"}, ...]
# Logs go to stderr.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REGISTRY="$REPO_ROOT/scripts/release/python-packages.json"

if [ ! -f "$REGISTRY" ]; then
  echo "ERROR: $REGISTRY not found" >&2
  exit 1
fi

jq -c '.[]' "$REGISTRY" | while read -r entry; do
  DIR=$(echo "$entry" | jq -r '.dir')
  BUILD_SYSTEM=$(echo "$entry" | jq -r '.build_system')
  PYPROJECT="$REPO_ROOT/$DIR/pyproject.toml"

  if [ ! -f "$PYPROJECT" ]; then
    echo "SKIP (no pyproject.toml): $DIR" >&2
    continue
  fi

  # Extract name and version based on build system
  if [ "$BUILD_SYSTEM" = "poetry" ]; then
    NAME=$(python3 -c "
import tomllib
with open('$PYPROJECT', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg['tool']['poetry']['name'])
" 2>/dev/null)
    VERSION=$(python3 -c "
import tomllib
with open('$PYPROJECT', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg['tool']['poetry']['version'])
" 2>/dev/null)
  else
    NAME=$(python3 -c "
import tomllib
with open('$PYPROJECT', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg['project']['name'])
" 2>/dev/null)
    VERSION=$(python3 -c "
import tomllib
with open('$PYPROJECT', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg['project']['version'])
" 2>/dev/null)
  fi

  if [ -z "$NAME" ] || [ -z "$VERSION" ]; then
    echo "SKIP (could not extract name/version): $DIR" >&2
    continue
  fi

  # Query PyPI JSON API for the published version
  PYPI_RESPONSE=$(curl -s "https://pypi.org/pypi/$NAME/json" 2>/dev/null || echo "")

  if [ -z "$PYPI_RESPONSE" ] || echo "$PYPI_RESPONSE" | jq -e '.message' &>/dev/null; then
    echo "NEW (unpublished): $NAME@$VERSION at $DIR" >&2
    echo "{\"name\":\"$NAME\",\"version\":\"$VERSION\",\"dir\":\"$DIR\",\"build_system\":\"$BUILD_SYSTEM\"}"
  else
    PUBLISHED_VERSION=$(echo "$PYPI_RESPONSE" | jq -r '.info.version')

    # Compare versions using Python's packaging library
    IS_NEWER=$(python3 -c "
from packaging.version import Version
try:
    local = Version('$VERSION')
    published = Version('$PUBLISHED_VERSION')
    print('true' if local > published else 'false')
except Exception:
    print('true' if '$VERSION' != '$PUBLISHED_VERSION' else 'false')
" 2>/dev/null || echo "false")

    if [ "$IS_NEWER" = "true" ]; then
      echo "CHANGED: $NAME $PUBLISHED_VERSION -> $VERSION at $DIR" >&2
      echo "{\"name\":\"$NAME\",\"version\":\"$VERSION\",\"dir\":\"$DIR\",\"build_system\":\"$BUILD_SYSTEM\"}"
    else
      echo "UP-TO-DATE: $NAME@$VERSION (published: $PUBLISHED_VERSION)" >&2
    fi
  fi
done | jq -sc '.'
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/release/detect-py-version-changes.sh
```

- [ ] **Step 3: Test locally**

```bash
bash scripts/release/detect-py-version-changes.sh 2>/dev/null | jq .
```
Expected: JSON array. Empty `[]` if all packages match their PyPI versions.

Check stderr:
```bash
bash scripts/release/detect-py-version-changes.sh >/dev/null
```
Expected: Lines showing each package's status.

Note: Requires Python 3.11+ for `tomllib` and `packaging` library (`pip install packaging` if not available). In the GitHub Actions workflow, `uv` provides the Python environment.

- [ ] **Step 4: Commit**

```bash
git add scripts/release/detect-py-version-changes.sh
git commit -m "Add Python version detection script for release automation"
```

---

### Task 4: Create GitHub Release Management Script

**Files:**
- Create: `scripts/release/create-or-update-release.sh`

This script creates a new GitHub Release or appends to an existing one for today's date. It handles the race condition between TypeScript and Python workflows.

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# scripts/release/create-or-update-release.sh
#
# Creates or updates a daily GitHub Release with published package info.
#
# Usage: ./create-or-update-release.sh <ecosystem> <packages-json>
#   ecosystem: "typescript" or "python"
#   packages-json: JSON array string of published packages
#     TS format:  [{"name":"@ag-ui/core","version":"0.0.49","path":"..."}]
#     Py format:  [{"name":"ag-ui-protocol","version":"0.1.15","dir":"..."}]
#
# Requires: gh CLI authenticated with contents:write permission
# Environment: DRY_RUN=true to skip actual release creation

set -euo pipefail

ECOSYSTEM="${1:?Usage: $0 <ecosystem> <packages-json>}"
PACKAGES_JSON="${2:?Usage: $0 <ecosystem> <packages-json>}"

TAG="release/$(date -u +%Y-%m-%d)"
TITLE="Release $(date -u +%Y-%m-%d)"
TIMESTAMP=$(date -u +%H:%M:%S)

# Build the section for this ecosystem
if [ "$ECOSYSTEM" = "typescript" ]; then
  SECTION="### TypeScript (npm) — published at ${TIMESTAMP} UTC\n"
  SECTION+="| Package | Version | Install |\n"
  SECTION+="|---------|---------|--------|\n"
  while read -r pkg; do
    NAME=$(echo "$pkg" | jq -r '.name')
    VERSION=$(echo "$pkg" | jq -r '.version')
    SECTION+="| $NAME | $VERSION | \`npm install ${NAME}@${VERSION}\` |\n"
  done < <(echo "$PACKAGES_JSON" | jq -c '.[]')
elif [ "$ECOSYSTEM" = "python" ]; then
  SECTION="### Python (PyPI) — published at ${TIMESTAMP} UTC\n"
  SECTION+="| Package | Version | Install |\n"
  SECTION+="|---------|---------|--------|\n"
  while read -r pkg; do
    NAME=$(echo "$pkg" | jq -r '.name')
    VERSION=$(echo "$pkg" | jq -r '.version')
    SECTION+="| $NAME | $VERSION | \`pip install ${NAME}==${VERSION}\` |\n"
  done < <(echo "$PACKAGES_JSON" | jq -c '.[]')
else
  echo "ERROR: Unknown ecosystem '$ECOSYSTEM'. Use 'typescript' or 'python'." >&2
  exit 1
fi

NEW_SECTION=$(echo -e "\n$SECTION")

if [ "${DRY_RUN:-false}" = "true" ]; then
  echo "DRY RUN: Would create/update release $TAG with:" >&2
  echo -e "$NEW_SECTION" >&2
  exit 0
fi

# Try to get existing release — retry logic for race condition
MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
  if gh release view "$TAG" &>/dev/null; then
    # Release exists — append our section
    EXISTING_BODY=$(gh release view "$TAG" --json body -q .body)
    UPDATED_BODY="${EXISTING_BODY}${NEW_SECTION}"
    gh release edit "$TAG" --notes "$UPDATED_BODY"
    echo "Updated existing release $TAG with $ECOSYSTEM packages" >&2
    exit 0
  else
    # Try to create new release
    BODY="## Packages Published${NEW_SECTION}"
    if gh release create "$TAG" --title "$TITLE" --notes "$BODY" 2>/dev/null; then
      echo "Created new release $TAG with $ECOSYSTEM packages" >&2
      exit 0
    else
      echo "Race condition on release creation (attempt $i/$MAX_RETRIES), retrying..." >&2
      sleep 2
    fi
  fi
done

echo "ERROR: Failed to create or update release after $MAX_RETRIES attempts" >&2
exit 1
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/release/create-or-update-release.sh
```

- [ ] **Step 3: Test with dry-run**

```bash
DRY_RUN=true bash scripts/release/create-or-update-release.sh typescript '[{"name":"@ag-ui/core","version":"0.0.49","path":"sdks/typescript/packages/core"}]'
```
Expected: Prints the release section to stderr with "DRY RUN" prefix.

```bash
DRY_RUN=true bash scripts/release/create-or-update-release.sh python '[{"name":"ag-ui-protocol","version":"0.1.15","dir":"sdks/python"}]'
```
Expected: Prints the Python release section.

- [ ] **Step 4: Commit**

```bash
git add scripts/release/create-or-update-release.sh
git commit -m "Add GitHub Release create-or-update script for release automation"
```

---

### Task 5: Create TypeScript Release Workflow

**Files:**
- Create: `.github/workflows/release-typescript.yml`

This is the main TypeScript release workflow. It triggers on push to `main` when TS-relevant files change, runs build + tests, detects version changes, publishes to npm, creates git tags, and creates/updates a GitHub Release.

- [ ] **Step 1: Create the workflow file**

```yaml
name: Release TypeScript Packages

on:
  push:
    branches: [main]
    paths:
      - "sdks/typescript/**"
      - "middlewares/**"
      - "integrations/*/typescript/**"
      - "integrations/community/*/typescript/**"
      - "pnpm-lock.yaml"
      - "pnpm-workspace.yaml"
      - "package.json"
      - ".github/workflows/release-typescript.yml"
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Run without publishing (build + detect only)"
        required: false
        type: boolean
        default: false

concurrency:
  group: release-typescript
  cancel-in-progress: false

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for git log in release notes

      - name: Install pnpm
        uses: pnpm/action-setup@v4
        with:
          version: 10.13.1

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "22"
          registry-url: "https://registry.npmjs.org"

      - name: Install protoc
        uses: arduino/setup-protoc@v3
        with:
          version: "25.x"
          repo-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Setup pnpm cache
        uses: actions/cache@v4
        with:
          path: ~/.local/share/pnpm/store
          key: ${{ runner.os }}-pnpm-store-${{ hashFiles('**/pnpm-lock.yaml') }}
          restore-keys: |
            ${{ runner.os }}-pnpm-store-

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build
        run: pnpm run build

      - name: Test
        run: pnpm run test

      - name: Detect version changes
        id: detect
        run: |
          CHANGED=$(bash scripts/release/detect-ts-version-changes.sh 2>detect-ts.log)
          cat detect-ts.log
          echo "packages=$CHANGED" >> "$GITHUB_OUTPUT"
          COUNT=$(echo "$CHANGED" | jq 'length')
          echo "count=$COUNT" >> "$GITHUB_OUTPUT"
          if [ "$COUNT" -gt 0 ]; then
            echo "## TypeScript packages to publish" >> $GITHUB_STEP_SUMMARY
            echo "$CHANGED" | jq -r '.[] | "- \(.name)@\(.version)"' >> $GITHUB_STEP_SUMMARY
          else
            echo "## No TypeScript version changes detected" >> $GITHUB_STEP_SUMMARY
          fi

      - name: Publish to npm
        if: steps.detect.outputs.count != '0' && (github.event.inputs.dry_run != 'true')
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
        run: |
          PACKAGES='${{ steps.detect.outputs.packages }}'
          echo "$PACKAGES" | jq -c '.[]' | while read -r pkg; do
            NAME=$(echo "$pkg" | jq -r '.name')
            VERSION=$(echo "$pkg" | jq -r '.version')
            PKG_PATH=$(echo "$pkg" | jq -r '.path')

            echo "Publishing $NAME@$VERSION from $PKG_PATH..."

            # Determine npm tag: alpha/beta/rc get their tag, stable gets latest
            if echo "$VERSION" | grep -qE '\-(alpha|beta|rc)'; then
              TAG=$(echo "$VERSION" | grep -oE '(alpha|beta|rc)')
              (cd "$PKG_PATH" && pnpm publish --no-git-checks --access public --tag "$TAG")
            else
              (cd "$PKG_PATH" && pnpm publish --no-git-checks --access public)
            fi

            echo "Published $NAME@$VERSION" >> $GITHUB_STEP_SUMMARY
          done

      - name: Create git tags
        if: steps.detect.outputs.count != '0' && (github.event.inputs.dry_run != 'true')
        run: |
          PACKAGES='${{ steps.detect.outputs.packages }}'
          echo "$PACKAGES" | jq -c '.[]' | while read -r pkg; do
            NAME=$(echo "$pkg" | jq -r '.name')
            VERSION=$(echo "$pkg" | jq -r '.version')
            TAG="${NAME}@${VERSION}"
            git tag "$TAG"
            echo "Tagged $TAG"
          done
          git push origin --tags

      - name: Create GitHub Release
        if: steps.detect.outputs.count != '0' && (github.event.inputs.dry_run != 'true')
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          bash scripts/release/create-or-update-release.sh typescript '${{ steps.detect.outputs.packages }}'

      - name: Dry-run summary
        if: steps.detect.outputs.count != '0' && (github.event.inputs.dry_run == 'true')
        env:
          DRY_RUN: "true"
        run: |
          echo "## Dry Run — would publish:" >> $GITHUB_STEP_SUMMARY
          echo '${{ steps.detect.outputs.packages }}' | jq -r '.[] | "- \(.name)@\(.version)"' >> $GITHUB_STEP_SUMMARY
          DRY_RUN=true bash scripts/release/create-or-update-release.sh typescript '${{ steps.detect.outputs.packages }}'
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/release-typescript.yml'))" && echo "YAML OK" || echo "YAML ERROR"
```
Expected: YAML OK

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release-typescript.yml
git commit -m "Add TypeScript release workflow (auto-publish to npm on version change)"
```

---

### Task 6: Create Python Release Workflow

**Files:**
- Create: `.github/workflows/release-python.yml`

- [ ] **Step 1: Create the workflow file**

```yaml
name: Release Python Packages

on:
  push:
    branches: [main]
    paths:
      - "sdks/python/**"
      - "integrations/*/python/**"
      - "integrations/community/*/python/**"
      - "scripts/release/python-packages.json"
      - ".github/workflows/release-python.yml"
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Run without publishing (detect + build only)"
        required: false
        type: boolean
        default: false

concurrency:
  group: release-python
  cancel-in-progress: false

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: ">=0.8.0"

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Poetry
        run: pip install poetry

      - name: Install packaging library
        run: pip install packaging

      - name: Detect version changes
        id: detect
        run: |
          CHANGED=$(bash scripts/release/detect-py-version-changes.sh 2>detect-py.log)
          cat detect-py.log
          echo "packages=$CHANGED" >> "$GITHUB_OUTPUT"
          COUNT=$(echo "$CHANGED" | jq 'length')
          echo "count=$COUNT" >> "$GITHUB_OUTPUT"
          if [ "$COUNT" -gt 0 ]; then
            echo "## Python packages to publish" >> $GITHUB_STEP_SUMMARY
            echo "$CHANGED" | jq -r '.[] | "- \(.name)@\(.version)"' >> $GITHUB_STEP_SUMMARY
          else
            echo "## No Python version changes detected" >> $GITHUB_STEP_SUMMARY
          fi

      - name: Build, test, and publish
        if: steps.detect.outputs.count != '0'
        env:
          UV_PUBLISH_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          PACKAGES='${{ steps.detect.outputs.packages }}'

          echo "$PACKAGES" | jq -c '.[]' | while read -r pkg; do
            NAME=$(echo "$pkg" | jq -r '.name')
            VERSION=$(echo "$pkg" | jq -r '.version')
            DIR=$(echo "$pkg" | jq -r '.dir')
            BUILD_SYSTEM=$(echo "$pkg" | jq -r '.build_system')

            echo "=== Processing $NAME@$VERSION ($BUILD_SYSTEM) ==="

            # Run in subshell to isolate directory changes
            (
              cd "$DIR"

              # Install dependencies
              if [ "$BUILD_SYSTEM" = "poetry" ]; then
                poetry install
              else
                uv sync
              fi

              # Run tests if configured
              TEST_CMD=$(python3 -c "
          import tomllib, sys
          cfg = tomllib.load(open('pyproject.toml', 'rb'))
          try:
              cmd = cfg['tool']['ag-ui']['scripts']['test']
              print(cmd)
          except KeyError:
              print('')
          " 2>/dev/null || echo "")

              if [ -n "$TEST_CMD" ]; then
                echo "Running tests: $TEST_CMD"
                if [ "$BUILD_SYSTEM" = "poetry" ]; then
                  poetry run $TEST_CMD
                else
                  uv run $TEST_CMD
                fi
              else
                echo "WARNING: No test script configured in [tool.ag-ui.scripts] for $NAME — skipping tests" >&2
                echo "⚠️ $NAME: no tests configured, skipping test step" >> $GITHUB_STEP_SUMMARY
              fi

              # Build
              if [ "$BUILD_SYSTEM" = "poetry" ]; then
                poetry build
              else
                uv build
              fi

              # Verify wheel permissions
              python3 -c "
          import zipfile, glob, sys
          whl = glob.glob('dist/*.whl')[0]
          print(f'Checking {whl}')
          bad = []
          for info in zipfile.ZipFile(whl).infolist():
              perms = (info.external_attr >> 16) & 0o777
              readable = perms & 0o444
              if not readable:
                  bad.append(info.filename)
          if bad:
              print(f'ERROR: {len(bad)} file(s) missing read permissions:', file=sys.stderr)
              for f in bad:
                  print(f'  - {f}', file=sys.stderr)
              sys.exit(1)
          print('All files have correct permissions.')
          "

              # Publish
              if [ "${{ github.event.inputs.dry_run }}" != "true" ]; then
                echo "Publishing $NAME@$VERSION to PyPI..."
                uv publish
                echo "Published $NAME@$VERSION" >> $GITHUB_STEP_SUMMARY
              else
                echo "DRY RUN: Would publish $NAME@$VERSION" >> $GITHUB_STEP_SUMMARY
              fi
            )
          done

      - name: Create git tags
        if: steps.detect.outputs.count != '0' && (github.event.inputs.dry_run != 'true')
        run: |
          PACKAGES='${{ steps.detect.outputs.packages }}'
          echo "$PACKAGES" | jq -c '.[]' | while read -r pkg; do
            NAME=$(echo "$pkg" | jq -r '.name')
            VERSION=$(echo "$pkg" | jq -r '.version')
            TAG="${NAME}@${VERSION}"
            git tag "$TAG"
            echo "Tagged $TAG"
          done
          git push origin --tags

      - name: Create GitHub Release
        if: steps.detect.outputs.count != '0' && (github.event.inputs.dry_run != 'true')
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          bash scripts/release/create-or-update-release.sh python '${{ steps.detect.outputs.packages }}'

      - name: Dry-run summary
        if: steps.detect.outputs.count != '0' && (github.event.inputs.dry_run == 'true')
        env:
          DRY_RUN: "true"
        run: |
          echo "## Dry Run — would publish:" >> $GITHUB_STEP_SUMMARY
          echo '${{ steps.detect.outputs.packages }}' | jq -r '.[] | "- \(.name)@\(.version)"' >> $GITHUB_STEP_SUMMARY
          DRY_RUN=true bash scripts/release/create-or-update-release.sh python '${{ steps.detect.outputs.packages }}'
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/release-python.yml'))" && echo "YAML OK" || echo "YAML ERROR"
```
Expected: YAML OK

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release-python.yml
git commit -m "Add Python release workflow (auto-publish to PyPI on version change)"
```

---

### Task 7: Push Branch and Validate

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feature/release-automation
```

- [ ] **Step 2: Run TypeScript dry-run via workflow_dispatch**

Go to https://github.com/ag-ui-protocol/ag-ui/actions/workflows/release-typescript.yml and trigger a manual run on `feature/release-automation` with `dry_run: true`.

Or via CLI:
```bash
gh workflow run release-typescript.yml --ref feature/release-automation -f dry_run=true
```

Expected: Workflow runs, builds, tests, detects version changes (likely 0 since all are current), and completes successfully. Check the step summary for output.

- [ ] **Step 3: Run Python dry-run via workflow_dispatch**

```bash
gh workflow run release-python.yml --ref feature/release-automation -f dry_run=true
```

Expected: Workflow runs, detects version changes, and completes. Check step summary.

**Important:** Verify that `claude-agent-sdk` (which uses `setuptools` as its build backend, not `uv_build`) builds correctly with `uv build`. If it fails, its `build_system` entry in `python-packages.json` may need to be changed, or the package may need a build system migration.

- [ ] **Step 4: Verify no errors in dry-run logs**

```bash
gh run list --workflow=release-typescript.yml --limit=1 --json conclusion,databaseId
gh run list --workflow=release-python.yml --limit=1 --json conclusion,databaseId
```

If any failures, check logs:
```bash
gh run view <run-id> --log-failed
```

- [ ] **Step 5: Fix any issues found in dry-run, re-commit, re-push, re-test**

Iterate until both dry-runs pass cleanly.

---

### Task 8: Final Commit Grouping and PR

After all dry-runs pass:

- [ ] **Step 1: Verify all files are committed**

```bash
git status
git log --oneline feature/release-automation ^main
```

Expected: Clean working tree, commits for each task above.

- [ ] **Step 2: Regroup commits by area of concern**

Reorganize into logical groups:
1. **Release scripts** — `python-packages.json`, `detect-ts-version-changes.sh`, `detect-py-version-changes.sh`, `create-or-update-release.sh`
2. **TypeScript release workflow** — `release-typescript.yml`
3. **Python release workflow** — `release-python.yml`
4. **Design spec** — `2026-03-25-release-automation-design.md`

- [ ] **Step 3: Create PR**

```bash
gh pr create --title "Add automated release to npm and PyPI on version change" --body "$(cat <<'EOF'
## Summary

Adds release automation for AG-UI:
- Automatically publishes TypeScript packages to npm when `package.json` versions are bumped on `main`
- Automatically publishes Python packages to PyPI when `pyproject.toml` versions are bumped on `main`
- Creates consolidated GitHub Releases with published package info
- Both workflows gate on tests passing and support manual dry-run via workflow_dispatch

## Design

See [Notion proposal](https://www.notion.so/32e3aa381852818db699c9f0ee12ba77) and `docs/superpowers/specs/2026-03-25-release-automation-design.md`.

## New files

| File | Purpose |
|------|---------|
| `.github/workflows/release-typescript.yml` | Auto-publish TS packages to npm |
| `.github/workflows/release-python.yml` | Auto-publish Python packages to PyPI |
| `scripts/release/python-packages.json` | Registry of publishable Python packages |
| `scripts/release/detect-ts-version-changes.sh` | Compare TS versions against npm |
| `scripts/release/detect-py-version-changes.sh` | Compare Python versions against PyPI |
| `scripts/release/create-or-update-release.sh` | Create/update daily GitHub Release |

## Prerequisites

- `NPM_TOKEN` secret must be added to the repository (for npm publishing)
- `PYPI_API_TOKEN` secret must exist in the repository (for PyPI publishing — already configured for existing `publish-python-package.yml`)

## Testing

- [x] Both workflows validated via dry-run on this branch
- [ ] First real release: bump one TS + one Python package version after merge

## Test plan

- [ ] Trigger both workflows manually in dry-run mode — verify they detect current versions correctly
- [ ] Merge to main with a version bump — verify publish + GitHub Release creation
- [ ] Push a fix commit after a failed release — verify re-trigger works automatically

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
