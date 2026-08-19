"""Narrow W&B Registry adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import wandb

from flight_delay.data.download import sha256_file
from flight_delay.modeling.release import ReleaseGuardError, verify_locked_files
from flight_delay.promotion.candidates import CandidateRecord
from flight_delay.promotion.policy import PromotionPolicy


class RegistryAdapterError(RuntimeError):
    """A query, precondition, mutation, or post-promotion verification failed."""


@dataclass(frozen=True)
class AliasState:
    registry_path: str
    alias: str
    version: str
    digest: str
    source_digest: str

    @property
    def immutable_identity(self) -> str:
        return f"{self.registry_path}:{self.version}@{self.digest}"


def _state(artifact: Any, registry_path: str, alias: str) -> AliasState:
    source = artifact.source_artifact if getattr(artifact, "is_link", False) else artifact
    return AliasState(
        registry_path=registry_path,
        alias=alias,
        version=artifact.version,
        digest=artifact.digest,
        source_digest=source.digest,
    )


class WandbRegistryAdapter:
    """Query exact Registry versions and mutate aliases only through supported linking."""

    def __init__(self, *, api_factory: Any = wandb.Api) -> None:
        self._api_factory = api_factory

    def _api(self) -> Any:
        return self._api_factory(timeout=60)

    def list_candidates(self, policy: PromotionPolicy) -> list[CandidateRecord]:
        try:
            artifacts = list(
                self._api().artifacts(
                    type_name=policy.artifact_type,
                    name=policy.registry_collection,
                    order="createdAt",
                    per_page=100,
                )
            )
        except Exception as error:
            raise RegistryAdapterError("unable to enumerate Registry candidates") from error
        candidates: list[CandidateRecord] = []
        for artifact in artifacts:
            try:
                with TemporaryDirectory(prefix="promotion-candidate-") as temporary:
                    root = Path(artifact.download(root=temporary))
                    lock = json.loads((root / "selection_lock.json").read_text(encoding="utf-8"))
                    verify_locked_files(root, lock)
                    bundle = root / "model_bundle"
                    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
                    metrics_payload = json.loads(
                        (bundle / "metrics_development.json").read_text(encoding="utf-8")
                    )
                    metrics = metrics_payload["metrics"]
                    model = joblib.load(bundle / "model.joblib")
                    threshold = float(
                        json.loads((bundle / "threshold.json").read_text(encoding="utf-8"))[
                            "threshold"
                        ]
                    )
                    source = (
                        artifact.source_artifact
                        if getattr(artifact, "is_link", False)
                        else artifact
                    )
                    period = lock["development_selection_period"]
                    required_integrity = (
                        "lineage_verified",
                        "schema_check_passed",
                        "leakage_check_passed",
                        "convergence_check_passed",
                        "serialization_check_passed",
                    )
                    raw = {
                        "candidate_id": f"{artifact.version}@{artifact.digest}",
                        "registry_path": policy.registry_collection,
                        "registry_version": artifact.version,
                        "registry_digest": artifact.digest,
                        "source_artifact_name": source.name,
                        "source_artifact_version": source.version,
                        "source_artifact_digest": source.digest,
                        "git_sha": metadata["reconstruction_git_sha"],
                        "dataset_artifact": metadata["dataset_artifact"],
                        "dataset_digest": metadata["dataset_digest"],
                        "feature_schema_sha256": sha256_file(bundle / "feature_schema.json"),
                        "evaluation_protocol": f"development-selection-v1:{period}",
                        "development_metrics": {
                            name: metrics[name] for name in policy.metric_names if name in metrics
                        },
                        "bundle_size_bytes": int(lock["bundle_size_bytes"]),
                        "release_eligible": all(
                            metrics.get(name) is True for name in required_integrity
                        ),
                        "lineage_verified": metrics.get("lineage_verified") is True,
                        "serialization_integrity": (
                            metrics.get("serialization_check_passed") is True
                        ),
                        "inference_compatible": (
                            hasattr(model, "predict_proba")
                            and 0 < threshold < 1
                            and lock.get("threshold") == threshold
                        ),
                        "aliases": [
                            alias for alias in artifact.aliases if not alias.startswith("v")
                        ],
                    }
                    candidates.append(
                        CandidateRecord.from_mapping(
                            raw,
                            forbidden_key_fragments=policy.forbidden_key_fragments,
                            forbidden_key_prefixes=policy.forbidden_key_prefixes,
                        )
                    )
            except (OSError, KeyError, ValueError, ReleaseGuardError) as error:
                raise RegistryAdapterError(
                    f"Registry candidate {getattr(artifact, 'version', 'unknown')} is invalid"
                ) from error
        return candidates

    def resolve_alias(self, registry_path: str, alias: str) -> AliasState | None:
        try:
            artifact = self._api().artifact(f"{registry_path}:{alias}")
        except wandb.errors.CommError as error:
            if "does not contain an artifact" in str(error) or "not found" in str(error).lower():
                return None
            raise RegistryAdapterError("unable to resolve Registry alias") from error
        except Exception as error:
            raise RegistryAdapterError("unable to resolve Registry alias") from error
        return _state(artifact, registry_path, alias)

    def promote(
        self,
        candidate: CandidateRecord,
        *,
        alias: str,
        expected_before: AliasState | None,
    ) -> AliasState:
        current = self.resolve_alias(candidate.registry_path, alias)
        if current is not None and current.immutable_identity == candidate.immutable_identity:
            return current
        if current != expected_before:
            raise RegistryAdapterError("Registry alias precondition changed concurrently")
        try:
            artifact = self._api().artifact(
                f"{candidate.registry_path}:{candidate.registry_version}"
            )
            if artifact.digest != candidate.registry_digest:
                raise RegistryAdapterError("candidate digest changed before promotion")
            linked = artifact.link(candidate.registry_path, aliases=[alias])
            linked.wait()
        except RegistryAdapterError:
            raise
        except Exception as error:
            raise RegistryAdapterError("Registry alias mutation failed") from error
        verified = self.resolve_alias(candidate.registry_path, alias)
        if verified is None or verified.immutable_identity != candidate.immutable_identity:
            raise RegistryAdapterError("Registry alias post-promotion verification failed")
        return verified
