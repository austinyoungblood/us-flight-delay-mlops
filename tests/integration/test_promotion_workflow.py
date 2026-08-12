from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_model_promotion_workflow_is_manual_and_separate_from_pr_ci() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/model-promotion.yml").read_text())
    triggers = workflow[True]
    assert set(triggers) == {"workflow_dispatch"}
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["dry_run"]["default"] is True
    assert inputs["target_alias"]["default"] == "production"
    assert workflow["permissions"] == {"contents": "read"}
    serialized = (ROOT / ".github/workflows/model-promotion.yml").read_text()
    assert "secrets.WANDB_API_KEY" in serialized
    assert "upload-artifact@v4" in serialized
    assert "push:" not in serialized
    assert "pull_request:" not in serialized


def test_model_promotion_workflow_fails_closed_without_wandb_key() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/model-promotion.yml").read_text())
    steps = workflow["jobs"]["promotion"]["steps"]
    preflight = next(step for step in steps if step.get("name") == "Require W&B API key")
    assert '[[ -z "${WANDB_API_KEY:-}" ]]' in preflight["run"]
    assert "::error title=Missing W&B credential::" in preflight["run"]
    assert "exit 1" in preflight["run"]


def test_stale_promotion_audit_is_removed_and_never_uploaded_after_failure() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/model-promotion.yml").read_text())
    steps = workflow["jobs"]["promotion"]["steps"]
    selection = next(
        step for step in steps if step.get("name") == "Select candidate and emit audit"
    )
    command_lines = [line.strip() for line in selection["run"].splitlines() if line.strip()]
    removal_index = command_lines.index("rm -f promotion_decision.json")
    live_command_index = next(
        index
        for index, line in enumerate(command_lines)
        if line.startswith("PYTHONPATH=src python scripts/promote_model.py")
    )
    assert live_command_index == removal_index + 1
    assert selection["id"] == "promotion"
    assert command_lines[live_command_index + 1] == "test -f promotion_decision.json"
    assert command_lines[live_command_index + 2] == (
        'echo "audit_produced=true" >> "$GITHUB_OUTPUT"'
    )

    upload = next(step for step in steps if step.get("name") == "Upload sanitized decision")
    assert "steps.promotion.outputs.audit_produced == 'true'" in upload["if"]
    assert "hashFiles('promotion_decision.json') != ''" in upload["if"]
    assert upload["with"]["path"] == "promotion_decision.json"
    assert upload["with"]["if-no-files-found"] == "error"


def test_normal_ci_has_no_registry_mutation_or_wandb_secret() -> None:
    serialized = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "promote_model.py" not in serialized
    assert "WANDB_API_KEY" not in serialized
