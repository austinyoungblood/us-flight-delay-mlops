# Local and live environment files

## Local Docker Compose

From the repository root, copy the non-secret local template to the ignored `.env` file and
populate only your W&B entity and API key:

```bash
cp deploy/env/local-compose.env.template .env
docker compose --env-file .env config --quiet
docker compose --env-file .env up -d --build
```

The local template uses the Compose/DynamoDB Local defaults. It is not a live-host template. Never
add AWS credentials, account identifiers, endpoints, or session values to it, and never commit the
populated `.env` copy.

## Live hosts

For a separately authorized live deployment, copy only the matching component template to
`/opt/us-flight-delay-mlops/<component>.env`, replace every placeholder on the target host, and set
mode `0600`. Never commit a completed host file.

The API host receives the W&B token. AWS temporary credentials and
`DYNAMODB_ENDPOINT_URL` are prohibited in all live host files: the API and monitor use
their EC2 instance profile. Set the actual live AWS region in the API and Monitor files; do not
reuse the local Compose region as deployment evidence. The Traveler host has neither AWS nor W&B
credentials.
