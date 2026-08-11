# DynamoDB infrastructure

`provision_dynamodb.py` is the non-destructive table provisioner shared by DynamoDB Local and a future
reviewed AWS deployment. It
creates the table only when absent and otherwise validates the exact contract. It never deletes or
recreates an incompatible table.

```bash
docker compose up -d dynamodb-local table-init
# Or, explicitly against an already-running local endpoint:
AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local \
  PYTHONPATH=src python infra/provision_dynamodb.py --endpoint-url http://127.0.0.1:8001
```

Defaults may be changed through `AWS_REGION` and `DYNAMODB_TABLE` or the `--region` and `--table`
arguments. The required default is:

- table `flight-delay-events`;
- PAY_PER_REQUEST billing;
- String partition key `pk`;
- GSI `event-date-created-at-index` with String `event_date` partition key, String `created_at` sort
  key, and ALL projection.

`DYNAMODB_ENDPOINT_URL`/`--endpoint-url` is development-only. When absent, boto3 retains its standard
production endpoint behavior; Brief 07 did not exercise that path. Starting the AWS Academy Learner
Lab or calling any AWS service is explicitly deferred to a separately reviewed increment.
