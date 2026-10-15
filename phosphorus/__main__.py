"""`python -m phosphorus` 入口。

与 `phosphorus.cli:main` 等价，便于在未安装 console script 的环境里直接运行。
"""

from __future__ import annotations

from phosphorus.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
