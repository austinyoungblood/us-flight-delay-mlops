.PHONY: build compose-check download-data format-check install lint log-dataset prepare-data \
	prepare-v3-data shell-check test train-candidate-a train-dummy validate validate-deployment \
	validate-evidence validate-final validate-v1 validate-v2 validate-v3 v3-dry-run

install:
	python -m pip install \
		-c requirements.lock \
		-c requirements-v1.lock \
		-c requirements-v2.lock \
		-e ".[dev,v1,v2]"

lint:
	ruff check .

format-check:
	ruff format --check .

test:
	WANDB_MODE=disabled pytest --cov=flight_delay --cov-branch --cov-report=term-missing --cov-fail-under=86

validate-v1:
	python scripts/validate_v1_protocol.py
	python scripts/run_v1_development.py
	python scripts/run_v1_december_qualification.py

validate-v2:
	python scripts/validate_v2_protocol.py
	python scripts/validate_v2_runtime.py
	python scripts/run_v2_development.py
	python scripts/run_v2_december_qualification.py

validate-v3:
	python scripts/validate_v3_protocol.py
	python scripts/run_v3_development.py
	python scripts/run_v3_december_qualification.py

validate-deployment:
	python scripts/validate_deployment_manifest.py

validate-evidence:
	python scripts/validate_evidence_manifest.py --require-files

shell-check:
	bash -n deploy/*.sh deploy/lib/*.sh

compose-check:
	docker compose --env-file deploy/env/local-compose.env.template config --quiet

# CI-aligned and offline: no AWS/W&B calls, data preparation, training, or publishing.
validate-final: export WANDB_MODE := disabled
validate-final: lint format-check test validate-v1 validate-v2 validate-v3 validate-deployment \
	validate-evidence shell-check compose-check

validate: validate-final

build:
	docker build -f services/api/Dockerfile -t flight-delay-api:scaffold .
	docker build -f services/user_ui/Dockerfile -t flight-delay-user-ui:scaffold .
	docker build -f services/monitor_ui/Dockerfile -t flight-delay-monitor-ui:scaffold .

download-data:
	python scripts/download_bts.py --config configs/base.yaml

prepare-data:
	python scripts/prepare_data.py --config configs/base.yaml

log-dataset:
	python scripts/log_dataset.py --config configs/base.yaml --wandb-mode online

train-dummy:
	python scripts/train.py --experiment configs/experiments/dummy.yaml --wandb-mode online

train-candidate-a:
	python scripts/train.py --experiment configs/experiments/candidate_a.yaml --wandb-mode online

prepare-v3-data:
	python scripts/prepare_v3_data.py --max-workers 8

v3-dry-run:
	python scripts/run_v3_development.py
