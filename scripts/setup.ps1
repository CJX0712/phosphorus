# Phosphorus 环境初始化（Windows PowerShell）
# 关键点：
#   1) pip 必须带 NO_PROXY='*'（系统注册表代理是 socks，pip 缺 PySocks 会崩）
#   2) 不依赖 MSVC —— C 内核走 `python -m ziglang cc`（O4）
param(
    [string]$Py = "python",
    [string]$Venv = ".venv",
    [string]$Group = "train,dev"
)

$ErrorActionPreference = "Stop"
$env:NO_PROXY = '*'
$env:no_proxy = '*'

Write-Host "[1/5] 创建虚拟环境: $Venv"
if (-not (Test-Path $Venv)) { & $Py -m venv $Venv }
$activate = Join-Path $Venv "Scripts\Activate.ps1"
& $activate

Write-Host "[2/5] 升级 pip"
& python -m pip install --upgrade pip

Write-Host "[3/5] 安装 phosphorus[$Group]"
& python -m pip install -e ".[$Group]"

Write-Host "[4/5] 验证 zig cc 编译链（O4）"
$zighas = & python -c "import ziglang; print('yes')" 2>$null
if ($zighas -eq "yes") {
    & python -m ziglang cc -O2 -o smoke_zig.exe scripts/smoke_zig.c
    if ($LASTEXITCODE -ne 0) { throw "zig cc 编译失败" }
    & .\smoke_zig.exe
    Remove-Item -Force smoke_zig.exe -ErrorAction SilentlyContinue
}
else {
    Write-Host "WARN: ziglang 未安装（可选），原生内核将回退到 NumPy 后端"
}

Write-Host "[5/5] 冒烟测试"
& python -c "import phosphorus; print('phosphorus', phosphorus.__version__)"
& python -m pytest -q

Write-Host "OK. 使用: .venv\Scripts\Activate.ps1 ; python -m phosphorus --help"
