#!/usr/bin/env bash
# One-time host setup for the Saarthi GPU server (Ubuntu 22.04/24.04).
# Review before running:  bash deploy/setup-ubuntu.sh
# It installs Docker + the NVIDIA container toolkit and creates deploy/.env.production.
# It does NOT install the NVIDIA driver (check `nvidia-smi` works first).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== 1/4  GPU driver"
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found. Install the NVIDIA driver first (e.g. 'sudo ubuntu-drivers install') and reboot." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== 2/4  Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER" || true

echo "== 3/4  NVIDIA container toolkit (lets containers use the GPU)"
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "== 4/4  Production env file"
if [ ! -f deploy/.env.production ]; then
  cp deploy/.env.production.example deploy/.env.production
  chmod 600 deploy/.env.production
  jwt="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
  adm="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  prx="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  sed -i "s|^JWT_SECRET=.*|JWT_SECRET=${jwt}|; s|^ADMIN_SECRET_TOKEN=.*|ADMIN_SECRET_TOKEN=${adm}|; s|^PROXY_SECRET=.*|PROXY_SECRET=${prx}|" deploy/.env.production
  echo "Created deploy/.env.production with fresh JWT_SECRET, ADMIN_SECRET_TOKEN and PROXY_SECRET."
else
  echo "deploy/.env.production already exists; left unchanged."
fi

echo
echo "Done. Log out and back in (docker group), fill in the remaining values in"
echo "deploy/.env.production, then follow docs/DEPLOYMENT.md from step 3."
echo "Check the GPU is visible to containers with:"
echo "  docker run --rm --gpus all ubuntu nvidia-smi"
