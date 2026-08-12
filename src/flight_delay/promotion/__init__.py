"""Policy-driven model selection and controlled Registry promotion."""

from flight_delay.promotion.audit import build_audit_record, write_audit_record
from flight_delay.promotion.candidates import CandidateMetadataError, CandidateRecord
from flight_delay.promotion.policy import PolicyError, PromotionPolicy, load_policy
from flight_delay.promotion.selector import SelectionResult, select_candidates
from flight_delay.promotion.wandb_registry import (
    AliasState,
    InMemoryRegistryAdapter,
    RegistryAdapterError,
    WandbRegistryAdapter,
)

__all__ = [
    "AliasState",
    "CandidateMetadataError",
    "CandidateRecord",
    "InMemoryRegistryAdapter",
    "PolicyError",
    "PromotionPolicy",
    "RegistryAdapterError",
    "SelectionResult",
    "WandbRegistryAdapter",
    "build_audit_record",
    "load_policy",
    "select_candidates",
    "write_audit_record",
]
