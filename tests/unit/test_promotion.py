from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import wandb
import yaml

from flight_delay.promotion import (
    AliasState,
    CandidateMetadataError,
    CandidateRecord,
    PolicyError,
    RegistryAdapterError,
    WandbRegistryAdapter,
    build_audit_record,
    load_policy,
    select_candidates,
    write_audit_record,
)

ROOT = Path(__file__).resolve().parents[2]


def policy():
    return load_policy(ROOT / "configs/promotion_policy.yaml")


def candidate(
    candidate_id: str = "candidate-a",
    *,
    version: str = "v0",
    digest: str = "a" * 32,
    average_precision: float = 0.28,
    f1: float = 0.37,
    recall: float = 0.71,
    roc_auc: float = 0.63,
    brier_score: float = 0.15,
    log_loss: float = 0.48,
    **updates,
) -> CandidateRecord:
    value = {
        "candidate_id": candidate_id,
        "registry_path": "wandb-registry-Model/us-flight-arrival-delay-15m",
        "registry_version": version,
        "registry_digest": digest,
        "source_artifact_name": f"flight-model:{version}",
        "source_artifact_version": version,
        "source_artifact_digest": digest,
        "git_sha": "1" * 40,
        "dataset_artifact": (
            "austin-youngblood-university-of-denver/us-flight-delay-mlops/"
            "flight-delay-bts-sampled:v0"
        ),
        "dataset_digest": "2ecdb5a6a60b23ed1ee1d603fb976516",
        "feature_schema_sha256": (
            "e319f9ff93823f8f1c3bb9043b1734e8995443779b1215661b0078dabe7a7a1e"
        ),
        "evaluation_protocol": "development-selection-v1:2025-11-16/2025-11-30",
        "development_metrics": {
            "average_precision": average_precision,
            "f1": f1,
            "recall": recall,
            "roc_auc": roc_auc,
            "brier_score": brier_score,
            "log_loss": log_loss,
        },
        "bundle_size_bytes": 555_000,
        "release_eligible": True,
        "lineage_verified": True,
        "serialization_integrity": True,
        "inference_compatible": True,
        "aliases": [],
    }
    value.update(updates)
    return CandidateRecord.from_mapping(value)


def state(item: CandidateRecord, alias: str = "production") -> AliasState:
    return AliasState(
        registry_path=item.registry_path,
        alias=alias,
        version=item.registry_version,
        digest=item.registry_digest,
        source_digest=item.source_artifact_digest,
    )


class InMemoryRegistryAdapter:
    """Deterministic Registry fake kept entirely within test support."""

    def __init__(
        self,
        candidates: list[CandidateRecord],
        aliases: dict[tuple[str, str], AliasState] | None = None,
        *,
        verification_failure: bool = False,
    ) -> None:
        self._candidates = list(candidates)
        self.aliases = dict(aliases or {})
        self.verification_failure = verification_failure
        self.mutations = 0

    def list_candidates(self, _policy: object) -> list[CandidateRecord]:
        return list(self._candidates)

    def resolve_alias(self, registry_path: str, alias: str) -> AliasState | None:
        return self.aliases.get((registry_path, alias))

    def promote(
        self,
        item: CandidateRecord,
        *,
        alias: str,
        expected_before: AliasState | None,
    ) -> AliasState:
        key = (item.registry_path, alias)
        current = self.aliases.get(key)
        if current is not None and current.immutable_identity == item.immutable_identity:
            return current
        if current != expected_before:
            raise RegistryAdapterError("Registry alias precondition changed concurrently")
        self.mutations += 1
        updated = AliasState(
            registry_path=item.registry_path,
            alias=alias,
            version=item.registry_version,
            digest=("wrong" if self.verification_failure else item.registry_digest),
            source_digest=item.source_artifact_digest,
        )
        self.aliases[key] = updated
        if updated.immutable_identity != item.immutable_identity:
            raise RegistryAdapterError("Registry alias post-promotion verification failed")
        return updated


def test_policy_is_versioned_and_prohibits_final_test_ranking(tmp_path: Path) -> None:
    loaded = policy()
    assert loaded.policy_version == "1.0.0"
    assert loaded.target_alias == "production"
    assert loaded.metric_names[0] == "average_precision"

    text = (ROOT / "configs/promotion_policy.yaml").read_text()
    text = text.replace("field: average_precision", "field: final_test_average_precision", 1)
    path = tmp_path / "policy.yaml"
    path.write_text(text)
    with pytest.raises(PolicyError, match="final-test"):
        load_policy(path)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), "v2", "schema version"),
        (("policy_version",), "", "non-empty string"),
        (("dry_run_default",), "yes", "must be boolean"),
        (("scope",), None, "scope must be a mapping"),
        (("selection_inputs", "development_validation_only"), False, "development/validation"),
        (("ranking",), [], "ranking must be non-empty"),
        (("ranking", 0, "field"), "", "ranking field"),
        (("ranking", 0, "direction"), "sideways", "direction is invalid"),
        (("mandatory_gates", "lineage_verified"), None, "candidate gates must be boolean"),
        (("mandatory_gates", "max_bundle_size_bytes"), 0, "must be positive"),
        (("allowed_target_aliases",), [], "allowed aliases"),
        (("scope", "registry_collection"), "", "scope or lineage is incomplete"),
        (("target_alias",), "candidate", "target alias is not allowed"),
        (("tie_breaking", "final_field"), "version", "final deterministic tie-break"),
    ],
)
def test_policy_validation_rejects_unsafe_shapes(
    tmp_path: Path, path: tuple[object, ...], value: object, message: str
) -> None:
    payload = yaml.safe_load((ROOT / "configs/promotion_policy.yaml").read_text(encoding="utf-8"))
    target: object = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(PolicyError, match=message):
        load_policy(policy_path)


def test_policy_root_and_ranking_items_must_be_mappings(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="promotion policy must be a mapping"):
        load_policy(policy_path)

    payload = yaml.safe_load((ROOT / "configs/promotion_policy.yaml").read_text(encoding="utf-8"))
    payload["ranking"][0] = "not-a-rule"
    policy_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(PolicyError, match="ranking rule must be a mapping"):
        load_policy(policy_path)


def test_one_candidate_promotes_and_incumbent_retains() -> None:
    item = candidate()
    promotion = select_candidates([item], policy(), incumbent_identity=None)
    retention = select_candidates([item], policy(), incumbent_identity=item.immutable_identity)
    assert promotion.outcome == "promote"
    assert promotion.winner == item
    assert retention.outcome == "retain_current"
    assert retention.requested_action == "none"


def test_multiple_candidates_rank_and_challenger_wins() -> None:
    incumbent = candidate("incumbent", average_precision=0.28)
    challenger = candidate("challenger", version="v1", digest="b" * 32, average_precision=0.29)
    result = select_candidates(
        [incumbent, challenger],
        policy(),
        incumbent_identity=incumbent.immutable_identity,
    )
    assert result.outcome == "promote"
    assert result.winner == challenger
    assert [entry["candidate_id"] for entry in result.ranking_explanation] == [
        "challenger",
        "incumbent",
    ]


def test_incumbent_wins_exact_metric_tie_before_candidate_id() -> None:
    first = candidate("a-first")
    incumbent = candidate("z-incumbent", version="v1", digest="b" * 32)
    result = select_candidates(
        [first, incumbent],
        policy(),
        incumbent_identity=incumbent.immutable_identity,
    )
    assert result.winner == incumbent
    assert result.outcome == "retain_current"


def test_candidate_id_is_final_tie_break_without_incumbent() -> None:
    alpha = candidate("alpha")
    beta = candidate("beta", version="v1", digest="b" * 32)
    result = select_candidates([beta, alpha], policy(), incumbent_identity=None)
    assert result.winner == alpha


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"dataset_digest": "wrong"}, "dataset digest"),
        ({"feature_schema_sha256": "wrong"}, "feature schema"),
        ({"evaluation_protocol": "different"}, "evaluation protocol"),
        ({"release_eligible": False}, "release_eligible"),
        ({"lineage_verified": False}, "lineage_verified"),
        ({"serialization_integrity": False}, "serialization_integrity"),
        ({"inference_compatible": False}, "inference_compatible"),
        ({"bundle_size_bytes": 10_000_001}, "bundle size"),
    ],
)
def test_policy_rejects_incompatible_candidate(updates: dict, reason: str) -> None:
    item = candidate(**updates)
    result = select_candidates([item], policy(), incumbent_identity=None)
    assert result.outcome == "no_eligible_candidate"
    assert reason in " ".join(result.rejected[0].reasons)


def test_missing_and_nonfinite_metrics_are_invalid_metadata() -> None:
    raw = candidate().__dict__ | {"aliases": []}
    missing = copy.deepcopy(raw)
    missing["development_metrics"].pop("average_precision")
    result = select_candidates([missing], policy(), incumbent_identity=None)
    assert result.outcome == "no_eligible_candidate"
    assert "missing" in result.rejected[0].reasons[0]

    invalid = copy.deepcopy(raw)
    invalid["development_metrics"]["average_precision"] = float("nan")
    result = select_candidates([invalid], policy(), incumbent_identity=None)
    assert result.outcome == "blocked_invalid_metadata"
    assert "finite" in result.rejected[0].reasons[0]


@pytest.mark.parametrize(
    "forbidden",
    [
        {"final_test_average_precision": 0.99},
        {"metadata": {"sealed_test_score": 0.99}},
        {"test_accuracy": 0.99},
    ],
)
def test_final_test_fields_are_rejected_as_selection_inputs(forbidden: dict) -> None:
    raw = candidate().__dict__ | {"aliases": []} | forbidden
    with pytest.raises(CandidateMetadataError, match="forbidden key"):
        CandidateRecord.from_mapping(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unknown": "value"}, "unknown fields"),
        ({"candidate_id": ""}, "identity fields"),
        ({"git_sha": "short"}, "full commit SHA"),
        ({"bundle_size_bytes": 0}, "must be positive"),
        ({"release_eligible": 1}, "gate fields must be boolean"),
        ({"development_metrics": {}}, "non-empty mapping"),
        ({"development_metrics": {"": 0.2}}, "metric names"),
        ({"development_metrics": {"average_precision": True}}, "must be numeric"),
        ({"aliases": ["production", ""]}, "aliases must be strings"),
    ],
)
def test_candidate_metadata_rejects_malformed_fields(mutation: dict, message: str) -> None:
    raw = candidate().__dict__ | {"aliases": []}
    raw.update(mutation)
    with pytest.raises(CandidateMetadataError, match=message):
        CandidateRecord.from_mapping(raw)


def test_candidate_metadata_requires_mapping_and_all_fields() -> None:
    with pytest.raises(CandidateMetadataError, match="must be a mapping"):
        CandidateRecord.from_mapping([])
    raw = candidate().__dict__ | {"aliases": []}
    raw.pop("dataset_digest")
    with pytest.raises(CandidateMetadataError, match="missing fields"):
        CandidateRecord.from_mapping(raw)


def test_candidate_aliases_are_deduplicated_and_audit_metrics_are_allowlisted() -> None:
    raw = candidate().__dict__ | {"aliases": ["staging", "production", "staging"]}
    item = CandidateRecord.from_mapping(raw)
    assert item.aliases == ("production", "staging")
    assert item.audit_view(("average_precision", "not-present"))["development_metrics"] == {
        "average_precision": 0.28
    }


def test_no_candidates_and_all_invalid_outcomes_are_explicit() -> None:
    empty = select_candidates([], policy(), incumbent_identity=None)
    assert empty.outcome == "no_eligible_candidate"
    invalid = candidate().__dict__ | {"aliases": [], "git_sha": "short"}
    blocked = select_candidates([invalid], policy(), incumbent_identity=None)
    assert blocked.outcome == "blocked_invalid_metadata"


def test_dry_run_causes_zero_mutation_and_apply_is_idempotent() -> None:
    item = candidate()
    adapter = InMemoryRegistryAdapter([item])
    before = adapter.resolve_alias(item.registry_path, "production")
    result = select_candidates([item], policy(), incumbent_identity=None)
    assert result.outcome == "promote"
    assert adapter.mutations == 0

    after = adapter.promote(item, alias="production", expected_before=before)
    assert after.immutable_identity == item.immutable_identity
    assert adapter.mutations == 1
    repeated = adapter.promote(item, alias="production", expected_before=after)
    assert repeated == after
    assert adapter.mutations == 1


def test_apply_surfaces_precondition_and_verification_failures() -> None:
    item = candidate()
    existing = candidate("old", version="v9", digest="9" * 32)
    adapter = InMemoryRegistryAdapter([item], {(item.registry_path, "production"): state(existing)})
    with pytest.raises(RegistryAdapterError, match="precondition"):
        adapter.promote(item, alias="production", expected_before=None)

    failing = InMemoryRegistryAdapter([item], verification_failure=True)
    with pytest.raises(RegistryAdapterError, match="verification"):
        failing.promote(item, alias="production", expected_before=None)


def test_sanitized_audit_contains_selection_evidence(tmp_path: Path) -> None:
    item = candidate()
    before = state(item)
    selection = select_candidates([item], policy(), incumbent_identity=item.immutable_identity)
    record = build_audit_record(
        mode="dry-run",
        policy=policy(),
        git_sha="2" * 40,
        target_alias="production",
        selection=selection,
        before=before,
        after=before,
        actual_action="none_dry_run",
        wandb_verified=True,
        workflow_identity={"github_run_id": "123"},
        timestamp=datetime(2026, 8, 12, tzinfo=UTC),
    )
    path = tmp_path / "promotion_decision.json"
    write_audit_record(path, record)
    saved = json.loads(path.read_text())
    assert saved["outcome"] == "retain_current"
    assert saved["selection_data_boundary"] == "development_validation_only"
    assert saved["before_alias"] == saved["after_alias"]
    assert "final_test" not in json.dumps(saved)
    assert "api_key" not in json.dumps(saved).lower()


def test_audit_writer_rejects_credentials(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="credential"):
        write_audit_record(tmp_path / "audit.json", {"wandb_api_key": "secret"})


class FakeLinked:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class FakeArtifact:
    def __init__(
        self,
        *,
        version: str,
        digest: str,
        source_digest: str | None = None,
    ) -> None:
        self.version = version
        self.digest = digest
        self.name = f"source:{version}"
        self.aliases = []
        self.is_link = source_digest is not None
        self.source_artifact = type(
            "SourceArtifact",
            (),
            {"digest": source_digest or digest, "name": self.name, "version": version},
        )()
        self.linked: FakeLinked | None = None

    def link(self, registry_path: str, aliases: list[str]) -> FakeLinked:
        assert registry_path == "wandb-registry-Model/us-flight-arrival-delay-15m"
        assert aliases == ["production"]
        self.linked = FakeLinked()
        return self.linked


class FakeApi:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses

    def artifact(self, path: str):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def wandb_adapter(api: FakeApi) -> WandbRegistryAdapter:
    return WandbRegistryAdapter(api_factory=lambda timeout: api)


def test_wandb_adapter_resolves_linked_alias_and_missing_alias() -> None:
    linked = FakeArtifact(version="v0", digest="a" * 32, source_digest="b" * 32)
    adapter = wandb_adapter(
        FakeApi([linked, wandb.errors.CommError("collection does not contain an artifact")])
    )
    resolved = adapter.resolve_alias(linked.name, "production")
    assert resolved is not None
    assert resolved.version == "v0"
    assert resolved.source_digest == "b" * 32
    assert adapter.resolve_alias(linked.name, "missing") is None


def test_wandb_adapter_surfaces_query_errors() -> None:
    adapter = wandb_adapter(FakeApi([RuntimeError("offline")]))
    with pytest.raises(RegistryAdapterError, match="resolve"):
        adapter.resolve_alias("collection", "production")

    class BrokenEnumeration:
        def artifacts(self, **kwargs):
            raise RuntimeError("offline")

    adapter = WandbRegistryAdapter(api_factory=lambda timeout: BrokenEnumeration())
    with pytest.raises(RegistryAdapterError, match="enumerate"):
        adapter.list_candidates(policy())


def test_wandb_adapter_promotes_exact_candidate_and_waits() -> None:
    item = candidate()
    old = FakeArtifact(version="v9", digest="9" * 32)
    target = FakeArtifact(version=item.registry_version, digest=item.registry_digest)
    verified = FakeArtifact(version=item.registry_version, digest=item.registry_digest)
    adapter = wandb_adapter(FakeApi([old, target, verified]))
    result = adapter.promote(
        item, alias="production", expected_before=state(candidate(version="v9", digest="9" * 32))
    )
    assert result.immutable_identity == item.immutable_identity
    assert target.linked is not None and target.linked.waited


def test_wandb_adapter_is_idempotent_and_checks_digest() -> None:
    item = candidate()
    current = FakeArtifact(version=item.registry_version, digest=item.registry_digest)
    adapter = wandb_adapter(FakeApi([current]))
    assert (
        adapter.promote(item, alias="production", expected_before=None).digest
        == item.registry_digest
    )

    old = FakeArtifact(version="v9", digest="9" * 32)
    drifted = FakeArtifact(version=item.registry_version, digest="wrong")
    adapter = wandb_adapter(FakeApi([old, drifted]))
    with pytest.raises(RegistryAdapterError, match="digest changed"):
        adapter.promote(
            item,
            alias="production",
            expected_before=state(candidate(version="v9", digest="9" * 32)),
        )
