.PHONY: build download-data format-check install lint log-dataset prepare-data test train-candidate-a train-dummy validate

install:
	python -m pip install -c requirements.lock -c requirements-v1.lock ".[dev,v1]"

lint:
	ruff check .

format-check:
	ruff format --check .

test:
	pytest --cov=flight_delay --cov-branch --cov-report=term-missing --cov-fail-under=82

validate: lint format-check test

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
