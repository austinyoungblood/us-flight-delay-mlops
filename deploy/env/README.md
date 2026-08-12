# Host environment files

Copy only the matching template to `/opt/us-flight-delay-mlops/<component>.env`, replace
placeholders on the target host, and set mode `0600`. Never commit the completed file.

The API host receives the W&B token. AWS temporary credentials and
`DYNAMODB_ENDPOINT_URL` are prohibited in all live host files: the API and monitor use
their EC2 instance profile. The traveler host has neither AWS nor W&B credentials.
