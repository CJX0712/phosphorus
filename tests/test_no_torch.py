"""质量红线：推理路径（core 分组）必须能在**没有 torch** 的环境里 import。

做法：在子进程里用 PYTHONPATH 前置一个「导入即抛错」的假 torch 包，
把真实的 torch 遮蔽掉；然后 import 全部运行时模块。任何一处 `import torch`
都会立刻炸出来，从而保证 `pip install -e .`（不装 [train]）也能推理。

这与 CI 的 `no-torch` job 等价，但不需要真的建一个无 torch 的 venv，
因此在装有 torch 的开发机上也能天天跑。
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_MODULES = [
    "phosphorus",
    "phosphorus.config",
    "phosphorus.packing",
    "phosphorus.gemm",
    "phosphorus.loader",
    "phosphorus.export",
    "phosphorus.sampling",
    "phosphorus.engine",
    "phosphorus.tokenizer",
    "phosphorus.data",
    "phosphorus.cli",
]


@pytest.fixture(scope="module")
def fake_torch(tmp_path_factory):
    """造一个 `import torch` 就抛错的假包，用于遮蔽真实 torch。"""
    d = tmp_path_factory.mktemp("no_torch")
    pkg = d / "torch"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        'raise RuntimeError("运行时不应 import torch：推理路径必须零 torch 依赖")\n',
        encoding="utf-8",
    )
    return d


def test_import_without_torch(fake_torch):
    """全部运行时模块在未装 torch 的环境下可 import。"""
    script = textwrap.dedent(
        """
        import sys
        mods = {mods!r}
        for m in mods:
            __import__(m)
        assert "torch" not in sys.modules, f"torch 被导入了: {{sorted(k for k in sys.modules if k.startswith('torch'))}}"
        import phosphorus
        print("OK", phosphorus.__version__)
        """
    ).format(mods=_MODULES)
    # Windows 控制台默认 GBK，子进程输出需显式指定 UTF-8，否则 text=True 解码失败
    env = {
        "PYTHONPATH": str(fake_torch),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PATH": "",
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
    )
    assert proc.returncode == 0, f"无 torch 导入失败:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout


def test_cli_help_without_torch(fake_torch):
    """`python -m phosphorus --help` 在无 torch 环境可用，且列出全部子命令。"""
    proc = subprocess.run(
        [sys.executable, "-m", "phosphorus", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path(__file__).resolve().parents[1]),
        env={
            "PYTHONPATH": str(fake_torch),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PATH": "",
        },
    )
    assert proc.returncode == 0, f"CLI 启动失败:\n{proc.stderr}"
    for cmd in ("tokenize", "train", "quantize", "export", "generate", "bench"):
        assert cmd in proc.stdout, f"子命令 {cmd} 未出现在 --help 中"
