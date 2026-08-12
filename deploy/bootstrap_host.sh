#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${DEPLOY_DRY_RUN:-0}" == "1" ]]; then
  printf 'dry-run bootstrap: detect Amazon Linux or Ubuntu; install Git/Python/Docker/curl; start Docker; create /opt/us-flight-delay-mlops with mode 750\n'
  exit 0
fi

[[ "${EUID}" -eq 0 ]] || { printf 'bootstrap failed: run as root\n' >&2; exit 1; }
case "$(uname -m)" in
  x86_64|aarch64) ;;
  *) printf 'bootstrap failed: unsupported architecture\n' >&2; exit 1 ;;
esac
source /etc/os-release
case "${ID}" in
  amzn)
    dnf install -y curl docker git python3
    ;;
  ubuntu)
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl docker.io git python3
    ;;
  *)
    printf 'bootstrap failed: unsupported operating system %s\n' "${ID}" >&2
    exit 1
    ;;
esac
systemctl enable --now docker
docker --version
curl --silent --show-error --output /dev/null https://ghcr.io/v2/
deploy_user="${SUDO_USER:-}"
if [[ -n "${deploy_user}" ]] && getent passwd "${deploy_user}" >/dev/null; then
  usermod -aG docker "${deploy_user}"
fi
install -d -o root -g docker -m 0750 /opt/us-flight-delay-mlops
printf 'bootstrap passed: Docker active and restricted deployment directory created\n'
