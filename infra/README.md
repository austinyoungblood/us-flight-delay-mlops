# DynamoDB infrastructure

`provision_dynamodb.py` is the non-destructive Brief 06 provisioner for AWS Academy Learner Lab. It
creates the table only when absent and otherwise validates the exact contract. It never deletes or
recreates an incompatible table.

```bash
cp .env.example .env
# Add current AWS Academy session credentials only to ignored .env.

PYTHONPATH=src python infra/provision_dynamodb.py --dry-run
PYTHONPATH=src python infra/provision_dynamodb.py
```

Defaults may be changed through `AWS_REGION` and `DYNAMODB_TABLE` or the `--region` and `--table`
arguments. The required default is:

- table `flight-delay-events`;
- PAY_PER_REQUEST billing;
- String partition key `pk`;
- GSI `event-date-created-at-index` with String `event_date` partition key, String `created_at` sort
  key, and ALL projection.

The 2026-08-10 external attempt was safely blocked before mutation: AWS returned
`UnrecognizedClientException` / `The security token included in the request is invalid` from
`DescribeTable`. Refresh the AWS Academy Learner Lab access key, secret key, and session token before
retrying. Do not commit credentials or provision any UI/EC2 resources as part of Brief 06.
