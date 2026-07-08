# NuGet package signing

The `AGUI.*` NuGet packages are **author-signed** during release so nuget.org
marks them as signed. Signing happens in the `publish-dotnet` job of
[`.github/workflows/publish-release.yml`](../../../.github/workflows/publish-release.yml),
in a provider-agnostic seam between **pack** and **push**:

```
pack  →  SIGN (this seam)  →  verify  →  push
```

The seam is **inert by default**. Until you set the `SIGNING_PROVIDER` variable
it does nothing and the release behaves exactly as before, so it is safe to
merge ahead of a certificate/account being provisioned.

> **Author vs. repository signature.** nuget.org always adds its own
> *repository* signature on push — that is automatic and unrelated to this.
> This seam adds the *author* signature, which is the one that carries an
> identity (see "Whose name shows up" below) and flips the package to "signed".
>
> **Not the same as `AGUI.snk`.** The `.snk` / `SignAssembly` machinery is
> assembly *strong-naming* (identity inside the DLLs). It is unrelated to NuGet
> author signing and is left untouched.

## Turning it on

Set a single variable on the **`nuget` environment** (Settings → Environments →
`nuget`), plus that provider's inputs:

| `SIGNING_PROVIDER` | Effect |
| --- | --- |
| _unset_ or `none` | No signing (default) |
| `signpath` | Sign via SignPath |
| `artifact-signing` | Sign via Azure Artifact Signing |

### Option A — SignPath

Free for OSS via the SignPath Foundation. The published signature reads
**"SignPath Foundation"**, not CopilotKit.

One-time SignPath-side setup (in the SignPath web app): create a **project**, a
**signing policy** (e.g. `release-signing`), and an **artifact configuration**
that signs the `*.nupkg` inside the uploaded artifact. Then wire:

| Name | Kind | Notes |
| --- | --- | --- |
| `SIGNING_PROVIDER` | var | `signpath` |
| `SIGNPATH_API_TOKEN` | **secret** | SignPath REST API token |
| `SIGNPATH_ORGANIZATION_ID` | var | SignPath organization ID |
| `SIGNPATH_PROJECT_SLUG` | var | e.g. `ag-ui-dotnet` |
| `SIGNPATH_SIGNING_POLICY_SLUG` | var | e.g. `release-signing` |
| `SIGNPATH_ARTIFACT_CONFIGURATION_SLUG` | var | artifact config that signs the nupkgs |

Mechanics: the packed `*.nupkg` are uploaded as a run artifact, SignPath signs
them remotely, and the signed files are downloaded back over the originals
(hence the job's `actions: read` permission).

### Option B — Azure Artifact Signing

Microsoft's managed service (formerly "Trusted Signing" / "Azure Code
Signing"), ~$9.99/mo. The signature reads **your validated org name** (e.g.
"CopilotKit"). Microsoft issues/rotates the certificate — you never hold a key;
you complete a one-time organization **identity validation** instead.

One-time Azure-side setup: create an Artifact Signing account + certificate
profile (Public Trust), complete identity validation, and configure an App
Registration with a **federated credential** for this repo/environment so the
job's OIDC token can log in (no client secret required). Then wire:

| Name | Kind | Notes |
| --- | --- | --- |
| `SIGNING_PROVIDER` | var | `artifact-signing` |
| `AZURE_CLIENT_ID` | **secret** | App Registration (client) ID — _already set on the `nuget` env (2026-07-01)_ |
| `AZURE_TENANT_ID` | **secret** | Directory (tenant) ID — _already set_ |
| `AZURE_SUBSCRIPTION_ID` | **secret** | Subscription containing the signing account — _already set_ |
| `ARTIFACT_SIGNING_ENDPOINT` | var | Regional endpoint, e.g. `https://eus.codesigning.azure.net` |
| `ARTIFACT_SIGNING_ACCOUNT` | var | Signing account name |
| `ARTIFACT_SIGNING_CERT_PROFILE` | var | Certificate profile name |

The three `AZURE_*` IDs are not really secret, but the OIDC federated identity
for this repo was pre-provisioned on the `nuget` environment as **secrets**
(2026-07-01), so the workflow reads them from `secrets`. Auth is federated
(OIDC) — no long-lived client secret is stored. Only the three
`ARTIFACT_SIGNING_*` values still need to be added (as vars) once the signing
account + certificate profile exist.

## Whose name shows up

| | SignPath (free OSS) | Azure Artifact Signing |
| --- | --- | --- |
| Signer shown in tooling | SignPath Foundation | your org, e.g. "CopilotKit" |
| Issuer (CA) | SignPath's CA partner | `Microsoft ID Verified CS …` |
| Cost | free | ~$9.99/mo |
| Gate to obtain | built-from-repo check | org identity validation |

## Signing already-published versions

NuGet.org versions are **immutable** — you cannot re-upload `0.0.3` as a signed
copy (the push step's `--skip-duplicate` will silently skip it). To ship signed
packages you must publish a **new version**:

1. Provision a provider above and set its vars/secrets.
2. Bump the shared `<VersionPrefix>` in
   [`sdks/dotnet/Directory.Build.props`](../Directory.Build.props)
   (e.g. `0.0.3` → `0.0.4`). One bump re-releases all five packages, signed.
3. (Optional) Unlist the old unsigned version on nuget.org so consumers move to
   the signed one.

## Provisioning checklist (Azure — primary path)

Ordered work to go from "seam merged" to "signed release." Ownership split:
Azure/infra owner does 1–3, a repo admin does 4, then anyone triggers 5.

1. **Signing account** — create a `Microsoft.CodeSigning/codeSigningAccounts`
   resource. ([Quickstart](https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart))
2. **Identity validation** — complete org validation (the multi-day gate); this
   is what stamps "CopilotKit" onto the certificate.
3. **Certificate profile** — create a **Public Trust** profile, and assign the
   **Trusted Signing Certificate Profile Signer** role to the existing App
   Registration (client `AZURE_CLIENT_ID`) so the OIDC login may sign.
   ([Roles tutorial](https://learn.microsoft.com/en-us/azure/trusted-signing/tutorial-assign-roles))
4. **Set the `nuget` environment variables** with the values from steps 1 & 3
   (the `AZURE_*` secrets already exist):
   ```bash
   R=ag-ui-protocol/ag-ui
   gh variable set ARTIFACT_SIGNING_ENDPOINT     --env nuget --repo $R --body "https://<region>.codesigning.azure.net"
   gh variable set ARTIFACT_SIGNING_ACCOUNT      --env nuget --repo $R --body "<signing-account-name>"
   gh variable set ARTIFACT_SIGNING_CERT_PROFILE --env nuget --repo $R --body "<certificate-profile-name>"
   gh variable set SIGNING_PROVIDER              --env nuget --repo $R --body "artifact-signing"
   ```
   > The `Preflight — validate signing configuration` step fails the run with an
   > explicit list if any of these are still missing, so a partial setup can't
   > publish a broken package.
5. **Trial + ship** — run `canary / publish` (scope `sdk-dotnet`,
   `dry_run=false`) → confirm signed on nuget.org → bump `VersionPrefix` for the
   stable signed release.

### SignPath (backup path)

If Azure stalls, set `SIGNING_PROVIDER=signpath` and the `SIGNPATH_*`
vars/secrets from the SignPath section above instead. Nothing else changes.
