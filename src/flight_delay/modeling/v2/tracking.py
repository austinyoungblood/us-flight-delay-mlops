"""Dependency-inverted, lazy W&B tracking for governed v2 execution."""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol


class TrackingRun(Protocol):
    id: str
    url: str

    def log(self, payload: dict[str, Any]) -> None: ...


class V2Tracker(Protocol):
    def start_run(self, *, name: str, group: str, metadata: dict[str, Any]) -> TrackingRun: ...


@dataclass
class NullRun(AbstractContextManager["NullRun"]):
    name: str
    metadata: dict[str, Any]
    id: str = "null"
    url: str = ""
    logged: list[dict[str, Any]] = field(default_factory=list)

    def __enter__(self) -> NullRun:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def log(self, payload: dict[str, Any]) -> None:
        self.logged.append(dict(payload))


@dataclass
class NullTracker:
    runs: list[NullRun] = field(default_factory=list)

    def start_run(self, *, name: str, group: str, metadata: dict[str, Any]) -> NullRun:
        run = NullRun(name=name, metadata={**metadata, "group": group})
        self.runs.append(run)
        return run


class WandbTracker:
    def __init__(self, *, entity: str, project: str) -> None:
        if not entity or not project:
            raise ValueError("online v2 tracking requires W&B entity and project")
        self.entity = entity
        self.project = project

    def start_run(self, *, name: str, group: str, metadata: dict[str, Any]) -> Any:
        wandb = importlib.import_module("wandb")
        return wandb.init(
            entity=self.entity,
            project=self.project,
            name=name,
            group=group,
            job_type="governed-v2-experiment",
            config=metadata,
            mode="online",
        )
