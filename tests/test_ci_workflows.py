r"""A CI check that can never pass is a silent no-op wearing a red badge.

WHY THIS EXISTS. This series argues that a gate which reports without ever acting is the dangerous
failure mode, and `tests/test_commit_gate_scope.py` pins the case where a scope regex stopped
matching so a hook printed `all green` having checked nothing. A permanently failing check is the
same defect with the sign flipped. It never turns green, so its colour carries no information, and
the reader learns to merge past red. That reflex is not hypothetical: four of this repository's
seven workflows failed on *every* run for weeks, and the habit of stepping over them nearly put an
`--admin` bypass on a pull request that needed no bypass at all.

The causes were not one family, which is why each is pinned separately below:

* `codeql` -- the analysis itself succeeded; only the SARIF upload was rejected, because uploading
  to code scanning needs a public repository or GitHub Advanced Security.
* `dependency-review` -- the action refuses to run at all under the same condition.
* `scorecard` -- a plain bug, independent of visibility: the job declared its own `permissions:`
  block, which *replaces* the workflow-level `read-all` rather than extending it, so `contents:
  read` was gone and `actions/checkout` could not clone. On a private repository an unauthorised
  clone is answered `404`, so the symptom read `fatal: repository ... not found` and looked like a
  missing repository rather than a permissions bug. On a public repository the same workflow would
  pass, because the clone succeeds anonymously -- the defect was latent until visibility flipped.
* `docs` -- `actions/deploy-pages` returns `404 Ensure GitHub Pages has been enabled`.
* `secret-scan` -- passed on pull requests and failed on its weekly schedule, because the action
  passes an empty `--since-commit`/`--branch` pair for `schedule` and `workflow_dispatch` only.
  Those are the runs that scan all history, where three detectors fire on non-secrets: every
  cassette's sha256 key trips one whose pattern is bare 64-hex; BigCodeBench task text containing
  `ftp://{ftp_user}:{ftp_password}@{ftp_server}` trips FTP; and `Lob`, whose pattern accepts
  `test_` plus 35 word characters, trips on this repository's own long test names -- 28 of them,
  which is exactly the count the failing run reported.

  `Lob` was missed on the first attempt, and the reason generalises: it was verified locally with
  `--no-verification`, which is NOT a superset of a verified scan, because a detector that only
  emits once verification succeeds is invisible to it. The local run reported zero findings while
  the dispatched run still failed with 28. Trust the dispatch, not the local dry run.

None of the fixes here changes the repository's visibility or its settings: the capability probe
reads what is actually available and the dependent steps skip with a reason, so the same workflows
start uploading the moment the repository is made public, with no further edit.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
_WORKFLOWS = ROOT / ".github" / "workflows"
_CAPABILITIES = ROOT / ".github" / "actions" / "repo-capabilities" / "action.yml"

#: Reference a job uses to pull in the capability probe. A local action, so the path must resolve.
_CAPABILITIES_REF = "./.github/actions/repo-capabilities"

#: Steps that cannot succeed without a public repository (or Advanced Security, which this
#: repository does not have). Matched against the action reference with the `@sha` stripped.
_NEEDS_PUBLIC = (
    "actions/dependency-review-action",
    "ossf/scorecard-action",
    "github/codeql-action/upload-sarif",
    # Attestations are available in public repositories on every plan, but in a private one only
    # under GitHub Enterprise Cloud. `release.yml` has never run, so this one was not red anywhere:
    # it would have failed on the first real tag, after the build and before the publish.
    "actions/attest-build-provenance",
)

#: Steps that cannot succeed unless GitHub Pages is enabled. `upload-pages-artifact` is
#: deliberately absent: it is an ordinary artifact upload and succeeds either way, so leaving it
#: ungated keeps the site build verified on every push.
_NEEDS_PAGES = ("actions/deploy-pages",)

#: Detector name -> the pattern that trips it, copied from the detector's own source at the pinned
#: trufflehog version. The exclusion list in `secret-scan.yml` is checked against what these
#: actually match in the tracked tree, in both directions, so the list cannot quietly grow into a
#: blanket suppression and cannot go stale.
#:
#: The corpus is every tracked file, not just the fixtures: `Lob` is tripped by test *names*, in
#: `tests/*.py`, and a fixtures-only sweep would have reported it as unjustified.
_TRIPWIRES = {
    # `\b((live|test)_[a-zA-Z0-9_]{35})\b` -- any test name with exactly 35 characters after
    # `test_`. 28 of them, over 19 files, which is exactly what the failing scan reported.
    "Lob": re.compile(r"\b(live|test)_[a-zA-Z0-9_]{35}\b"),
    # Bare 64-hex: every cassette is keyed by the sha256 of its request.
    "PhraseAccessToken": re.compile(r"\b[0-9a-f]{64}\b"),
    # BigCodeBench task text, whose credentials are f-string placeholders.
    "FTP": re.compile(r"ftp://[^\s\"'\\]*:[^\s\"'\\]*@"),
}


def _load(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workflows() -> list[tuple[str, dict]]:
    found = [(p.name, _load(p)) for p in sorted(_WORKFLOWS.glob("*.yml"))]
    assert found, f"no workflows under {_WORKFLOWS}"
    return found


def _triggers(wf: dict) -> dict:
    """Return the `on:` block.

    PyYAML follows YAML 1.1, where `on` is a boolean, so a GitHub workflow parses with `True` as
    the key and `wf["on"]` raises `KeyError` on every file in the directory.
    """
    if "on" in wf:
        return wf["on"]
    return wf[True]


def _steps(wf: dict) -> list[tuple[str, dict, dict]]:
    """Every step as (job name, job, step)."""
    return [
        (name, job, step)
        for name, job in wf.get("jobs", {}).items()
        for step in job.get("steps", [])
    ]


def _action(step: dict) -> str:
    """The action reference with any `@sha` pin stripped."""
    return str(step.get("uses", "")).split("@")[0]


def _gate(job: dict, step: dict) -> str:
    """The condition guarding a step: its own `if`, plus the job's, since either suffices."""
    return f"{step.get('if', '')} {job.get('if', '')}"


def _capability_outputs() -> list[str]:
    """The probe's declared outputs, so the assertions below name no capability of their own."""
    declared = sorted(_load(_CAPABILITIES).get("outputs", {}))
    assert declared, f"{_CAPABILITIES} declares no outputs"
    return declared


def _grants_contents_read(permissions: object) -> bool:
    if permissions is None:
        # No block at all leaves the default token scope, which can read the repository.
        return True
    if isinstance(permissions, str):
        return permissions in {"read-all", "write-all"}
    if isinstance(permissions, dict):
        return permissions.get("contents") in {"read", "write"}
    return False


def _excluded_detectors() -> set[str]:
    for _name, wf in _workflows():
        for _job_name, _job, step in _steps(wf):
            args = str(step.get("with", {}).get("extra_args", ""))
            found = re.search(r"--exclude-detectors=([\w,]+)", args)
            if found:
                return {d for d in found.group(1).split(",") if d}
    return set()


def _tripped_detectors() -> set[str]:
    """Which detectors the tracked tree actually trips, measured rather than asserted.

    Scans the files trufflehog would see, read from `git ls-files` rather than a directory list, so
    adding a fixture or a test file cannot leave this sweep behind. The working tree is the
    boundary: trufflehog scans git *history*, so a detector tripped only by a since-deleted file
    would not show up here. The authoritative check is the dispatched run, which is why
    `workflow_dispatch` exists.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    paths = [p for p in listed.stdout.split("\0") if p]
    assert len(paths) > 100, f"git ls-files gave only {len(paths)} paths; the sweep is vacuous"

    remaining = dict(_TRIPWIRES)
    tripped: set[str] = set()
    read = 0
    for rel in paths:
        if not remaining:
            break
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        read += 1
        for name, pattern in list(remaining.items()):
            if pattern.search(text):
                tripped.add(name)
                del remaining[name]
    assert read, "read no tracked files; the sweep checked nothing"
    return tripped


# --------------------------------------------------------------------------------------------
# The capability probe itself
# --------------------------------------------------------------------------------------------


def test_the_capability_probe_exists_and_is_a_composite_action() -> None:
    assert _CAPABILITIES.exists(), f"{_CAPABILITIES} is missing"
    action = _load(_CAPABILITIES)
    assert action["runs"]["using"] == "composite"
    assert _capability_outputs() == ["pages", "public"]


def test_every_local_action_reference_resolves_on_disk() -> None:
    """The lesson from the pre-commit hook: a path in CI config is code, and it can go stale."""
    checked = 0
    for name, wf in _workflows():
        for job_name, _job, step in _steps(wf):
            ref = _action(step)
            if not ref.startswith("./"):
                continue
            target = ROOT / ref[2:]
            assert (target / "action.yml").exists() or (target / "action.yaml").exists(), (
                f"{name}:{job_name} uses {ref}, which has no action.yml on disk"
            )
            checked += 1
    assert checked, "no workflow references a local action; this assertion checked nothing"


def test_the_probe_is_used_only_after_a_checkout_in_the_same_job() -> None:
    """A local action lives in the repository, so it cannot be resolved before the clone."""
    checked = 0
    for name, wf in _workflows():
        for job_name, job in wf.get("jobs", {}).items():
            seen_checkout = False
            for step in job.get("steps", []):
                ref = _action(step)
                if ref == "actions/checkout":
                    seen_checkout = True
                elif ref == _CAPABILITIES_REF:
                    assert seen_checkout, (
                        f"{name}:{job_name} uses {ref} before any actions/checkout step"
                    )
                    checked += 1
    assert checked, f"no job uses {_CAPABILITIES_REF}; this assertion checked nothing"


# --------------------------------------------------------------------------------------------
# No step may depend on a capability this repository might not have
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("actions", "output"),
    [(_NEEDS_PUBLIC, "public"), (_NEEDS_PAGES, "pages")],
    ids=["public", "pages"],
)
def test_capability_dependent_steps_are_gated(actions: tuple[str, ...], output: str) -> None:
    assert output in _capability_outputs()
    gated = 0
    for name, wf in _workflows():
        for job_name, job, step in _steps(wf):
            if _action(step) not in actions:
                continue
            assert f"outputs.{output}" in _gate(job, step), (
                f"{name}:{job_name} runs {_action(step)}, which needs `{output}`, "
                f"under condition {_gate(job, step)!r}"
            )
            gated += 1
    assert gated, f"no step in any workflow depends on `{output}`; the pattern list is stale"


def test_the_codeql_upload_is_conditional_rather_than_the_analysis() -> None:
    """Keep scanning when the results cannot be uploaded; only the upload needs the capability."""
    wf = _load(_WORKFLOWS / "codeql.yml")
    analyze = [s for _j, _job, s in _steps(wf) if _action(s) == "github/codeql-action/analyze"]
    assert len(analyze) == 1, f"expected one analyze step, found {len(analyze)}"
    step = analyze[0]
    assert "if" not in step, "the analysis must run unconditionally; gate its upload instead"
    upload = str(step["with"]["upload"])
    assert "outputs.public" in upload, f"analyze upload is not conditional: {upload!r}"
    assert "'always'" in upload and "'never'" in upload, upload


def test_unuploadable_codeql_results_are_kept_as_an_artifact() -> None:
    """Otherwise gating the upload would discard the findings rather than publish them elsewhere."""
    wf = _load(_WORKFLOWS / "codeql.yml")
    analyze = next(s for _j, _job, s in _steps(wf) if _action(s) == "github/codeql-action/analyze")
    artifact = [s for _j, _job, s in _steps(wf) if _action(s) == "actions/upload-artifact"]
    assert len(artifact) == 1, "the SARIF must be kept when it cannot be uploaded"
    step = artifact[0]
    assert "outputs.public" in str(step.get("if", "")), "keep the artifact only when not uploading"
    assert str(step["with"]["path"]) == str(analyze["with"]["output"]), (
        "the artifact path must be the directory analyze wrote its SARIF to"
    )


# --------------------------------------------------------------------------------------------
# The scorecard bug: job-level `permissions` replaces, it does not extend
# --------------------------------------------------------------------------------------------


def test_a_job_that_attests_declares_the_permission_the_upload_needs() -> None:
    """`id-token: write` signs the attestation; `attestations: write` is what stores it."""
    checked = 0
    for name, wf in _workflows():
        for job_name, job in wf.get("jobs", {}).items():
            if not any(_action(s).startswith("actions/attest") for s in job.get("steps", [])):
                continue
            effective = job["permissions"] if "permissions" in job else wf.get("permissions")
            granted = effective == "write-all" or (
                isinstance(effective, dict) and effective.get("attestations") == "write"
            )
            assert granted, (
                f"{name}:{job_name} attests a build with permissions {effective!r}, which does "
                f"not grant `attestations: write`"
            )
            checked += 1
    assert checked, "no job attests a build; the action prefix is stale"


def test_every_job_that_checks_out_can_read_a_private_repository() -> None:
    for name, wf in _workflows():
        for job_name, job in wf.get("jobs", {}).items():
            if not any(_action(s) == "actions/checkout" for s in job.get("steps", [])):
                continue
            effective = job["permissions"] if "permissions" in job else wf.get("permissions")
            assert _grants_contents_read(effective), (
                f"{name}:{job_name} checks out with permissions {effective!r}, which does not "
                f"grant `contents: read`; a job-level block replaces the workflow-level one"
            )


# --------------------------------------------------------------------------------------------
# The secret scan must be quiet for the right reason, and provable on demand
# --------------------------------------------------------------------------------------------


def test_the_secret_scan_excludes_exactly_the_detectors_its_fixtures_trip() -> None:
    """Both directions: no unexplained suppression, and no known false positive left to fire."""
    assert _excluded_detectors() == _tripped_detectors(), (
        "the exclusion list and the fixtures disagree; every excluded detector must be one the "
        "tracked fixtures demonstrably trip, and every detector they trip must be excluded"
    )


def test_the_secret_scan_can_be_run_on_demand() -> None:
    """`schedule` and `workflow_dispatch` are the only full-history runs, so a fix to the
    history-only failure is unverifiable for a week without a manual trigger."""
    triggers = _triggers(_load(_WORKFLOWS / "secret-scan.yml"))
    assert "workflow_dispatch" in triggers, sorted(triggers)
    assert "schedule" in triggers, sorted(triggers)
