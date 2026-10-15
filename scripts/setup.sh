#!/usr/bin/env bash
# Phosphorus 环境初始化（Linux / macOS）
# 关键点：
#   1) pip 必须带 NO_PROXY='*'（本机的系统代理是 socks，pip 缺 PySocks 会崩）
#   2) 不依赖系统编译器 —— C 内核走 `python -m ziglang cc`（O4）
set -euo pipefail

PY="${PY:-python3}"
VENV="${VENV:-.venv}"
GROUP="${GROUP:-train,dev}"

export NO_PROXY='*'
export no_proxy='*'

echo "[1/5] 创建虚拟环境: $VENV"
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[2/5] 升级 pip"
python -m pip install --upgrade pip

echo "[3/5] 安装 phosphorus[$GROUP]"
python -m pip install -e ".[$GROUP]"

echo "[4/5] 验证 zig cc 编译链（O4）"
if python -c "import ziglang" 2>/dev/null; then
  python -m ziglang cc -O2 -o /tmp/phos_smoke scripts/smoke_zig.c
  /tmp/phos_smoke
  rm -f /tmp/phos_smoke
else
  echo "WARN: ziglang 未安装（可选），原生内核将回退到 NumPy 后端"
fi

echo "[5/5] 冒烟测试"
python -c "import phosphorus; print('phosphorus', phosphorus.__version__)"
python -m pytest -q

echo "OK. 使用: source $VENV/bin/activate && python -m phosphorus --help"
