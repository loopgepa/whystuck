r"""A CI check that can never pass is a silent no-op wearing a red badge.

WHY THIS EXISTS. This series argues that a gate which reports without ever acting is the dangerous
failure mode, and `tests/test_commit_gate_scope.py` pins the case where a scope regex stopped
matching so a hook printed `all green` having checked nothing. A permanently failing check is the
same defect with the sign flipped. It never turns green, so its colour carries no information, and
the reader learns to merge past red. That reflex is not hypothetical: four of this repository's
seven workflows failed on *every* run for weeks, and the habit of stepping over them nearly put an
`--admin` bypass on a pull request that needed no bypass at all.

The four causes were not one family, which is why each is pinned separately below:

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
  Those are the runs that scan all history, where every cassette's sha256 key trips a detector
  whose pattern is bare 64-hex, and BigCodeBench task text containing
  `ftp://{ftp_user}:{ftp_password}@{ftp_server}` trips the FTP detector.

None of the fixes here changes the repository's visibility or its settings: the capability probe
reads what is actually available and the dependent steps skip with a reason, so the same workflows
start uploading the moment the repository is made public, with no further edit.
"""

from __future__ import annotations

import pathlib
import re

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
)

#: Steps that cannot succeed unless GitHub Pages is enabled. `upload-pages-artifact` is
#: deliberately absent: it is an ordinary artifact upload and succeeds either way, so leaving it
#: ungated keeps the site build verified on every push.
_NEEDS_PAGES = ("actions/deploy-pages",)

#: Detector name -> the fixture shape that trips it. The exclusion list in `secret-scan.yml` is
#: checked against what these actually match in the tracked fixtures, in both directions, so the
#: list cannot quietly grow into a blanket suppression and cannot go stale if the fixtures change.
_TRIPWIRES = {
    "PhraseAccessToken": re.compile(r"\b[0-9a-f]{64}\b"),
    "FTP": re.compile(r"ftp://[^\s\"'\\]*:[^\s\"'\\]*@"),
}

_FIXTURES = ROOT / "tests" / "fixtures" / "cassettes"


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
    """Which detectors the tracked fixtures actually trip, measured rather than asserted."""
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in sorted(_FIXTURES.rglob("*.json"))
    )
    assert text, f"no fixtures under {_FIXTURES}"
    return {name for name, pattern in _TRIPWIRES.items() if pattern.search(text)}


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
