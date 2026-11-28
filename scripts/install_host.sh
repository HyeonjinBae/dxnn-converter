#!/usr/bin/env bash
# 변환 PC 설치 스크립트 (Linux x86-64 / WSL2 Ubuntu, Python 3.10+)
#   ./scripts/install_host.sh                 # dxconv + DX-COM 2.5.0
#   DXCOM_VERSION=2.4.1 ./scripts/install_host.sh   # DX-COM 버전 지정 (가상환경 하나에 한 버전)
#   PYTHON=python3.11 ./scripts/install_host.sh     # python 지정
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
DXCOM_VERSION=${DXCOM_VERSION:-2.5.0}

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  echo "오류: DX-COM 은 Linux x86-64 전용입니다 (이 PC: $(uname -s) $(uname -m))." >&2
  echo "      Linux x86-64 PC 또는 Windows WSL2 Ubuntu 에서 실행하세요 (docs/windows_wsl2.md)." >&2
  exit 1
fi

if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "오류: Python 3.10 이상이 필요합니다 ($PY: $("$PY" --version 2>&1))." >&2
  echo "      다른 python 을 쓰려면: PYTHON=python3.12 ./scripts/install_host.sh" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

echo "DX-COM ${DXCOM_VERSION} 설치"
pip install "dx-com==${DXCOM_VERSION}"

dxconv env || true
