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

Brief 06 implements the backend data plane. FastAPI reads the serving alias only from
`release/release_decision.json`, verifies the exact W&B Registry version/digest and every locked
artifact hash, loads the immutable model once during lifespan, and fails closed if Registry or
DynamoDB is unavailable. This release remains `staging`: Registry `v0`, digest
`865ddd18f6debd44f24a79fc71739f2a`, threshold `0.1840285229739868`.

All six backend endpoints, bounded inference-only caching, conditional prediction/feedback writes,
active-model metadata, Decimal-safe DynamoDB serialization, and the idempotent table provisioner are
implemented and hermetically tested. The real W&B loader passed; the AWS Academy smoke is blocked
because `DescribeTable` rejected the configured session token as invalid. No table mutation, UI,
monitoring dashboard, EC2 deployment, or demo seeding occurred. See the
[model-selection report](docs/model-selection-report.md) and
[remediation report](docs/model-remediation-report.md), the
[final-test report](docs/final-test-report.md), and the
[API/DynamoDB status report](docs/api-dynamodb-status.md), and the
[detailed status ledger](docs/implementation-status.md).

## Architecture summary

The system boundary has three independently deployable services. FastAPI now consumes the exact
`staging` release and owns validation, inference, route context, caching, and required event
persistence. The traveler and monitoring Streamlit processes remain honest placeholders for the next
brief. The [architecture document](docs/architecture.md) marks current and future components.

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

docker build -f services/api/Dockerfile -t flight-delay-api:brief06 .
docker build -f services/user_ui/Dockerfile -t flight-delay-user-ui:brief06 .
docker build -f services/monitor_ui/Dockerfile -t flight-delay-monitor-ui:brief06 .
```

## API and DynamoDB runtime

Copy `.env.example` to ignored `.env` and supply `WANDB_API_KEY`, `WANDB_ENTITY`, current AWS Academy
session credentials, `AWS_REGION`, and `DYNAMODB_TABLE`. Do not set a model alias: the committed
release decision is deliberately the sole serving control plane.

```bash
PYTHONPATH=src python infra/provision_dynamodb.py --dry-run
PYTHONPATH=src python infra/provision_dynamodb.py
uvicorn services.api.app.main:app --host 127.0.0.1 --port 8000
```

The required table is `flight-delay-events`, PAY_PER_REQUEST, with String partition key `pk` and ALL
projection GSI `event-date-created-at-index` (`event_date`, `created_at`). `/health` returns 200 only
when both the verified Registry runtime and DynamoDB are ready; otherwise it returns structured 503
without a local-model fallback. Successful prediction responses are returned only after their unique
event is persisted.

Implemented endpoints are `GET /health`, `GET /model-info`, `POST /predict`,
`GET /route-reliability`, `GET /predictions/{id}`, and `POST /feedback/{id}`. Interactive docs are at
`/docs`. User and monitoring placeholders remain on ports 8501 and 8502 under Compose.

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
│   ├── persistence/
│   └── serving/
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

## Next reviewed phase

The exact next increment is **user Streamlit + separate DynamoDB-backed monitoring dashboard
consuming this API/data plane**. EC2 deployment, CloudWatch setup, drift automation, and demo-data
seeding remain outside Brief 06.
