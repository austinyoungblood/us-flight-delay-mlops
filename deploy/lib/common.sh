#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  printf 'deployment failed: %s\n' "$1" >&2
  exit 1
}

manifest_value() {
  local command="$1"
  local component="$2"
  local manifest="$3"
  python3 "${DEPLOY_ROOT}/deploy/read_manifest.py" \
    "${command}" "${component}" --manifest "${manifest}"
}

validate_env_file() {
  local component="$1"
  local env_file="$2"
  local manifest="$3"
  [[ -f "${env_file}" ]] || fail "environment file is missing"
  [[ "$(stat -c '%a' "${env_file}")" == "600" ]] || fail "environment file mode must be 600"
  if grep -Eq '^(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|DYNAMODB_ENDPOINT_URL)=' "${env_file}"; then
    fail "cloud environment file contains a forbidden AWS credential or endpoint variable"
  fi
  local expected actual
  expected="$(manifest_value env-names "${component}" "${manifest}")"
  actual="$(awk -F= '/^[A-Z][A-Z0-9_]*=/{print $1}' "${env_file}" | sort -u | \
    python3 -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
  [[ "${actual}" == "${expected}" ]] || fail "environment file names disagree with the manifest"
  if grep -Eq '=(__SET_AT_RUNTIME__|CHANGE_ME|)$' "${env_file}"; then
    fail "environment file contains an unset placeholder"
  fi
}

docker_command() {
  if docker info >/dev/null 2>&1; then
    printf 'docker'
  elif sudo -n docker info >/dev/null 2>&1; then
    printf 'sudo docker'
  else
    fail "Docker is not available to the current user"
  fi
}

deploy_component() {
  local component="$1"
  local container_name="$2"
  local health_path="$3"
  shift 3
  local manifest="${DEPLOY_ROOT}/deploy/deployment_manifest.json"
  local env_file=""
  while (($#)); do
    case "$1" in
      --manifest) manifest="$2"; shift 2 ;;
      --env-file) env_file="$2"; shift 2 ;;
      *) fail "unknown argument: $1" ;;
    esac
  done
  [[ -n "${env_file}" ]] || fail "--env-file is required"
  local image port
  image="$(manifest_value image "${component}" "${manifest}")"
  port="$(manifest_value port "${component}" "${manifest}")"
  [[ "${image}" =~ ^ghcr\.io/[a-z0-9_.-]+/[a-z0-9_.-]+@sha256:[0-9a-f]{64}$ ]] || \
    fail "image must be pinned by GHCR digest"
  validate_env_file "${component}" "${env_file}" "${manifest}"

  if [[ "${DEPLOY_DRY_RUN:-0}" == "1" ]]; then
    printf 'dry-run validated component=%s container=%s image=%s port=%s health=%s\n' \
      "${component}" "${container_name}" "${image}" "${port}" "${health_path}"
    return 0
  fi

  local docker_cmd
  docker_cmd="$(docker_command)"
  read -r -a docker_parts <<<"${docker_cmd}"
  "${docker_parts[@]}" pull "${image}"
  "${docker_parts[@]}" stop "${container_name}" >/dev/null 2>&1 || true
  "${docker_parts[@]}" rm "${container_name}" >/dev/null 2>&1 || true
  "${docker_parts[@]}" run -d --name "${container_name}" --restart unless-stopped \
    --env-file "${env_file}" -p "${port}:${port}" "${image}" >/dev/null
  [[ "$("${docker_parts[@]}" inspect --format '{{.State.Running}}' "${container_name}")" == "true" ]] || \
    fail "named container is not running"
  curl --fail --silent --show-error --retry 12 --retry-delay 5 \
    "http://127.0.0.1:${port}${health_path}" >/dev/null
  if [[ "${component}" == "api" ]]; then
    curl --fail --silent --show-error "http://127.0.0.1:${port}/model-info" | \
      python3 -c 'import json,sys; value=json.load(sys.stdin); allowed=("registry_path","serving_alias","registry_version","registry_digest","bundle_digest"); print(json.dumps({key:value[key] for key in allowed}, sort_keys=True))'
  fi
  printf 'deployment passed component=%s image=%s\n' "${component}" "${image}"
}
