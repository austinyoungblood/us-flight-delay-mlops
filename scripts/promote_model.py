#!/usr/bin/env python3
"""Validate policy or execute a controlled W&B Registry promotion decision."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import wandb
from dotenv import load_dotenv

from flight_delay.promotion import (
    RegistryAdapterError,
    WandbRegistryAdapter,
    build_audit_record,
    load_policy,
    select_candidates,
    write_audit_record,
)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _workflow_identity() -> dict[str, str]:
    names = (
        "GITHUB_ACTIONS",
        "GITHUB_REPOSITORY",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_WORKFLOW",
        "GITHUB_SHA",
    )
    return {name.lower(): os.environ[name] for name in names if os.environ.get(name)}


def _log_wandb_audit(path: Path, record: dict[str, Any]) -> str:
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise RuntimeError("WANDB_ENTITY and WANDB_PROJECT are required to log the audit run")
    with wandb.init(
        entity=entity,
        project=project,
        job_type="model-promotion-audit",
        config={
            "policy_version": record["policy"]["version"],
            "policy_sha256": record["policy"]["sha256"],
            "mode": record["mode"],
            "target_alias": record["target_alias"],
            "selection_data_boundary": record["selection_data_boundary"],
        },
    ) as run:
        run.summary["outcome"] = record["outcome"]
        run.summary["actual_action"] = record["actual_action"]
        run.summary["wandb_verified"] = record["wandb_verification"]["verified"]
        artifact = wandb.Artifact("model-promotion-decision", type="promotion-audit")
        artifact.add_file(str(path), name="promotion_decision.json")
        run.log_artifact(artifact)
        return run.url


def execute(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    target_alias = args.target_alias or policy.target_alias
    policy.validate_target_alias(target_alias)
    adapter = WandbRegistryAdapter()
    candidates = adapter.list_candidates(policy)
    if args.candidate_version:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.registry_version == args.candidate_version
        ]
    before = adapter.resolve_alias(policy.registry_collection, target_alias)
    incumbent_identity = before.immutable_identity if before else None
    selection = select_candidates(
        candidates,
        policy,
        incumbent_identity=incumbent_identity,
    )
    after = before
    actual_action = "none_dry_run" if args.mode == "dry-run" else "none"
    verified = bool(
        selection.winner is not None
        and before is not None
        and before.immutable_identity == selection.winner.immutable_identity
    )
    if args.mode == "apply" and selection.outcome == "promote":
        assert selection.winner is not None
        after = adapter.promote(
            selection.winner,
            alias=target_alias,
            expected_before=before,
        )
        actual_action = "alias_moved"
        verified = after.immutable_identity == selection.winner.immutable_identity
    elif args.mode == "apply" and selection.outcome == "retain_current":
        actual_action = "none_already_current"
        verified = True

    record = build_audit_record(
        mode=args.mode,
        policy=policy,
        git_sha=_git_sha(),
        target_alias=target_alias,
        selection=selection,
        before=before,
        after=after,
        actual_action=actual_action,
        wandb_verified=verified,
        workflow_identity=_workflow_identity(),
    )
    write_audit_record(args.output, record)
    run_url = _log_wandb_audit(args.output, record) if args.log_wandb_run else None
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": args.mode,
                "outcome": selection.outcome,
                "actual_action": actual_action,
                "winner": selection.winner.immutable_identity if selection.winner else None,
                "audit_path": str(args.output),
                "wandb_verified": verified,
                "wandb_run_url": run_url,
            },
            sort_keys=True,
        )
    )
    return 0 if selection.winner is not None and verified else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--policy", type=Path, default=Path("configs/promotion_policy.yaml"))
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-policy")
    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    run.add_argument("--target-alias", default=None)
    run.add_argument("--candidate-version", default=None)
    run.add_argument("--output", type=Path, default=Path("promotion_decision.json"))
    run.add_argument("--log-wandb-run", action="store_true")
    return result


def main() -> int:
    load_dotenv()
    args = parser().parse_args()
    try:
        policy = load_policy(args.policy)
        if args.command == "validate-policy":
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "policy_version": policy.policy_version,
                        "policy_sha256": policy.policy_sha256,
                        "target_alias": policy.target_alias,
                    },
                    sort_keys=True,
                )
            )
            return 0
        return execute(args)
    except (OSError, ValueError, RuntimeError, RegistryAdapterError, wandb.Error) as error:
        print(f"model promotion failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
