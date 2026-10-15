# Phosphorus · 常用命令入口（全部命令不依赖系统编译器）
# 用法： make <target>
PY      ?= python
PYTEST  ?= $(PY) -m pytest
RUFF    ?= $(PY) -m ruff
ZIG     ?= $(PY) -m ziglang
NO_PROXY ?= *
export NO_PROXY no_proxy

.PHONY: help install install-dev test lint fmt check build-native smoke-zig \
        tokenize train quantize export generate bench ci clean

help: ## 显示帮助
	@echo "Phosphorus Makefile"
	@echo "  make install      安装 core 依赖（推理即可用）"
	@echo "  make install-dev  安装 core + train + dev"
	@echo "  make smoke-zig    验证 zig cc 编译链（O4 前置）"
	@echo "  make build-native 编译 C99 原生内核（T4）"
	@echo "  make tokenize     训练 BPE 并编码语料"
	@echo "  make train        CPU 训练（fp32 或 qat-hgf）"
	@echo "  make quantize     导出 .phos"
	@echo "  make generate     文本生成"
	@echo "  make bench        基准评测"
	@echo "  make ci           lint + test 全量门禁"

install: ## 安装 core 分组
	$(PY) -m pip install -e .

install-dev: ## 安装 core + train + dev 分组
	$(PY) -m pip install -e ".[train,dev]"

smoke-zig: ## 验证 zig cc 可用（O4 零编译器依赖构建）
	$(ZIG) cc -O2 -o smoke_zig scripts/smoke_zig.c
	./smoke_zig
	@rm -f smoke_zig smoke_zig.exe

build-native: ## 编译 C99 原生内核（T4 落地）
	$(PY) -c "from phosphorus.kernels.native import build; print(build.ensure_library())"

tokenize: ## 训练 BPE 分词器并生成 token 缓存
	$(PY) -m phosphorus tokenize --corpus builtin --vocab-size 8192 --out artifacts/tokenizer.json

train: ## 默认配置训练（fp32 基线）
	$(PY) -m phosphorus train --mode fp32 --steps 300 --out ckpt/tiny-fp32.pt

quantize: ## 导出 .phos
	$(PY) -m phosphorus quantize --in ckpt/tiny-fp32.pt --out artifacts/tiny.phos --hgf-rank 4

generate: ## 文本生成
	$(PY) -m phosphorus generate --model artifacts/tiny.phos --prompt "To be, or not to be"

bench: ## 基准评测
	$(PY) -m phosphorus bench --model artifacts/tiny.phos --backend fp32

test: ## 跑全部测试
	$(PYTEST)

lint: ## ruff 检查
	$(RUFF) check .

fmt: ## ruff 格式化
	$(RUFF) format .

check: lint test ## 全局门禁

ci: check ## CI 等价门禁

clean: ## 清理临时产物
	rm -rf artifacts ckpt .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
