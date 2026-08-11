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

Brief 02 adds a verified official BTS Reporting Carrier downloader, canonical source/processed
manifests, deterministic chronological Parquet splits, and online W&B tracking for a Dummy baseline
and Candidate A. The online [W&B project](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops)
contains dataset artifact
[`flight-delay-bts-sampled:v0`](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/dataset/flight-delay-bts-sampled/v0),
the [Dummy run](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/g0cnsglm),
and the [Candidate A run](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/7mt7qz71).

The final test split is prepared and versioned but remains evaluation-sealed. No model is loaded by
the API, and Registry promotion, database work, monitoring implementation, AWS, and deployment have
not started. See [the detailed status ledger](docs/implementation-status.md).

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
pytest --cov=flight_delay --cov-branch --cov-report=term-missing --cov-fail-under=80

docker build -f services/api/Dockerfile -t flight-delay-api:scaffold .
docker build -f services/user_ui/Dockerfile -t flight-delay-user-ui:scaffold .
docker build -f services/monitor_ui/Dockerfile -t flight-delay-monitor-ui:scaffold .
```

Run all local service shells with `docker compose up --build`. The API health endpoint is
`http://localhost:8000/health`; user and monitoring placeholders use ports 8501 and 8502.

## Data and experiment workflow

The official archive family is
`On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip` from the
[BTS PREZIP directory](https://transtats.bts.gov/PREZIP/). The workflow downloads January 2025
through May 2026, applies a deterministic class-stratified cap of 75,000 eligible rows/month with
seed 42, and writes these half-open windows:

- train: `[2025-01-01, 2025-11-01)` — 750,000 rows;
- validation: `[2025-11-01, 2026-01-01)` — 150,000 rows;
- sealed test: `[2026-01-01, 2026-06-01)` — 375,000 rows.

From the repository root:

```bash
make download-data
make prepare-data

cp .env.example .env
chmod 600 .env
# Edit .env: set WANDB_API_KEY and WANDB_ENTITY; keep WANDB_PROJECT=us-flight-delay-mlops.
set -a
source .env
set +a

make log-dataset
make train-dummy
make train-candidate-a
make validate
```

`WANDB_API_KEY` is secret and must exist only in the shell environment or ignored `.env`; never put
it in configuration, source, manifests, logs, or reports. `WANDB_ENTITY`, `WANDB_PROJECT`, and
`WANDB_MODE` are non-secret configuration. Tests use a fake adapter/disabled mode and need no
network or credentials.

Validation-only results at threshold 0.5:

| Run | Accuracy | Precision | Recall | F1 | Average precision | ROC-AUC | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy prior | 0.76334 | 0 | 0 | 0 | 0.23666 | 0.5 | 0.180962 | 0.548087 |
| Candidate A | 0.569967 | 0.302541 | 0.625961 | 0.407923 | 0.320719 | 0.622293 | 0.245282 | 0.684221 |

The model artifacts are
[`flight-delay-model:v0`](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/model/flight-delay-model/v0)
(Dummy) and
[`flight-delay-model:v1`](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/model/flight-delay-model/v1)
(Candidate A). Neither is linked to W&B Registry.

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

- **Next, only after Brief 02 review:** Candidate B, validation threshold selection, frozen model
  bundle, one-time final-test evaluation, and W&B Registry `staging`/`production` aliases.
- **Later:** FastAPI inference/cache, DynamoDB event persistence, traveler workflow,
  database-backed monitoring/drift, AWS deployment, and operational evidence.

These are future plans, not completed features.
