# U.S. Flight Delay MLOps

A production-oriented course project for predicting whether a scheduled U.S. domestic flight will
arrive at least 15 minutes late. BTS treats arrivals less than 15 minutes after schedule as on time.

## Prediction framing

The prediction is made **before departure** and may use only scheduled flight information: carrier,
route, calendar, scheduled departure/arrival, scheduled elapsed time, and distance. Actual departure,
taxi, airborne, arrival, cancellation/diversion outcome, and delay-cause fields are forbidden. The
central leakage guard rejects both explicitly forbidden and merely unapproved model fields.

This project will estimate risk from historical scheduled-flight data. It will not provide live
flight status or a guarantee about an individual flight.

## Current implementation status

Brief 01 establishes a tested foundation: contracts, leakage-safe in-memory preprocessing,
chronological splitting, deterministic monthly sampling, historical route reliability aggregation,
a health-only FastAPI app, two Streamlit placeholders, CI, and three container definitions.

No dataset has been downloaded. No model has been trained or loaded. No W&B or AWS resources exist,
and no prediction, persistence, feedback, or production monitoring endpoint is implemented. See
[the detailed status ledger](docs/implementation-status.md).

## Architecture summary

The intended final system has three independently deployed services: a traveler Streamlit UI that
calls FastAPI, a FastAPI service that loads the W&B Registry `production` artifact and writes every
event to DynamoDB, and a monitoring Streamlit app that reads DynamoDB separately. Offline training
will log datasets, runs, and model artifacts to W&B. The [architecture document](docs/architecture.md)
marks current and future components explicitly.

## Python setup and validation

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements.lock ".[dev]"

python --version
ruff check .
ruff format --check .
pytest --cov=flight_delay --cov-report=term-missing

docker build -f services/api/Dockerfile -t flight-delay-api:scaffold .
docker build -f services/user_ui/Dockerfile -t flight-delay-user-ui:scaffold .
docker build -f services/monitor_ui/Dockerfile -t flight-delay-monitor-ui:scaffold .
```

Run all local service shells with `docker compose up --build`. The API health endpoint is
`http://localhost:8000/health`; user and monitoring placeholders use ports 8501 and 8502.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── configs/base.yaml
├── data/README.md
├── deploy/
├── docs/
├── evidence/
├── infra/
├── scripts/
├── services/
│   ├── api/
│   ├── user_ui/
│   └── monitor_ui/
├── src/flight_delay/
│   ├── contracts/
│   ├── data/
│   ├── features/
│   ├── modeling/
│   ├── monitoring/
│   └── persistence/
├── tests/unit/
├── tests/integration/
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Data and secret policy

Raw/processed BTS data, trained models, W&B files, AWS credentials, `.env`, caches, and large
artifacts must never be committed. Tests use only small in-memory fixtures. `.env.example` contains
names and non-secret defaults only.

## Planned reviewed phases

- **Next:** official BTS downloader, dataset checksum manifest, W&B dataset artifact, and
  Dummy/Candidate A training.
- **Later:** model selection and W&B Registry, FastAPI inference/cache, DynamoDB event persistence,
  traveler workflow, database-backed monitoring and drift, three-EC2 deployment, and evidence.

These are plans, not completed features.
