#!/usr/bin/env bash
# 변환 PC 설치 스크립트 (Linux x86-64 / WSL2 Ubuntu, Python 3.10+)
#   ./scripts/install_host.sh                 # dxconv + DX-COM 2.5.0
#   DXCOM_VERSION=2.4.1 ./scripts/install_host.sh   # DX-COM 버전 지정 (가상환경 하나에 한 버전)
#   PYTHON=python3.11 ./scripts/install_host.sh     # python 지정
# macOS 는 DX-COM 을 설치하지 않고 docker 이미지로 실행한다 (README 설치 참고).
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
DXCOM_VERSION=${DXCOM_VERSION:-2.5.0}

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 이상이 필요합니다"'

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
  echo "DX-COM ${DXCOM_VERSION} 설치"
  pip install "dx-com==${DXCOM_VERSION}"
else
  echo "안내: DX-COM 은 Linux x86-64 전용입니다. 이 PC($(uname -s) $(uname -m))는 docker 이미지로 실행하세요."
fi

dxconv env || true
