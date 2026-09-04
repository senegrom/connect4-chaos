# Launch the Modal loop driver detached at Idle.
# Usage: scripts/launch-modal-loop.ps1 -Init <model on Volume> -Gen <first gen> [-K] [-Games] [-Steps] [-Batch] [-Lr]
#        [-Window] [-MinNew] [-Sims] [-ArenaEvery] [-ArenaLag] [-Shapes] [-TargetSims] [-TargetShare]
# The Modal-environment interpreter comes from C4_MODAL_PYTHON (default D:\PyEnv\modal\Scripts\python.exe).
param([string]$Init, [int]$Gen, [int]$K = 3, [int]$Games = 4096, [int]$Steps = 6000,
      [int]$Batch = 1024, [double]$Lr = 4e-4, [int]$Window = 4000000, [int]$MinNew = 2000000,
      [int]$Sims = 0, [int]$ArenaEvery = 5, [int]$ArenaLag = 5, [string]$Shapes = 'all',
      [int]$TargetSims = 0, [double]$TargetShare = 0.25)
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUTF8 = '1'
$root = if ($env:C4_NEURAL_ROOT) { $env:C4_NEURAL_ROOT } else { 'E:\tmp-claude\connect4\neural' }
$python = if ($env:C4_MODAL_PYTHON) { $env:C4_MODAL_PYTHON } else { 'D:\PyEnv\modal\Scripts\python.exe' }
# Refuse to start a second driver: two loops spawn double the H100 work and
# can publish an older generation over a newer one.
$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'neural\.modal_loop|[\/]modal-loop\.py' })
if ($running.Count -gt 0) {
    "refusing to launch: loop driver already running (pid $($running[0].ProcessId)); stop it first with $root\modal-loop.stop"
    exit 1
}
if (Test-Path "$root\modal-loop.stop") { Remove-Item "$root\modal-loop.stop" }
$args = @('-m', 'neural.modal_loop', $Init, "$Gen", "$K", "$Games", "$Steps", "$Batch", "$Lr", "$Window", "$MinNew", "$Sims", "$ArenaEvery", "$ArenaLag", $Shapes, "$TargetSims", "$TargetShare")
$p = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory (Split-Path $PSScriptRoot -Parent) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root\modal-loop.stdout" -RedirectStandardError "$root\modal-loop.stderr"
$p.PriorityClass = 'Idle'
"loop driver pid $($p.Id) $($p.PriorityClass) init=$Init gen=$Gen K=$K"
