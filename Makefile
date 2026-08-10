.PHONY: install lint format-check test validate build

install:
	python -m pip install -c requirements.lock ".[dev]"

lint:
	ruff check .

format-check:
	ruff format --check .

test:
	pytest --cov=flight_delay --cov-report=term-missing

validate: lint format-check test

build:
	docker build -f services/api/Dockerfile -t flight-delay-api:scaffold .
	docker build -f services/user_ui/Dockerfile -t flight-delay-user-ui:scaffold .
	docker build -f services/monitor_ui/Dockerfile -t flight-delay-monitor-ui:scaffold .
