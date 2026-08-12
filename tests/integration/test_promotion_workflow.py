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


def test_normal_ci_has_no_registry_mutation_or_wandb_secret() -> None:
    serialized = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "promote_model.py" not in serialized
    assert "WANDB_API_KEY" not in serialized
