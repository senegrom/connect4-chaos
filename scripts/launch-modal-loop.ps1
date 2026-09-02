# Launch the Modal loop driver detached at Idle.
# Usage: scripts/launch-modal-loop.ps1 <init model name on Volume> <first gen> [K] [games] [steps] [batch] [lr] [window] [minNew]
param([string]$Init, [int]$Gen, [int]$K = 3, [int]$Games = 4096, [int]$Steps = 6000,
      [int]$Batch = 1024, [double]$Lr = 4e-4, [int]$Window = 4000000, [int]$MinNew = 2000000)
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUTF8 = '1'
$root = if ($env:C4_NEURAL_ROOT) { $env:C4_NEURAL_ROOT } else { 'E:\tmp-claude\connect4\neural' }
if (Test-Path "$root\modal-loop.stop") { Remove-Item "$root\modal-loop.stop" }
$args = @('-m', 'neural.modal_loop', $Init, "$Gen", "$K", "$Games", "$Steps", "$Batch", "$Lr", "$Window", "$MinNew")
$p = Start-Process -FilePath 'D:\PyEnv\modal\Scripts\python.exe' -ArgumentList $args -WorkingDirectory (Split-Path $PSScriptRoot -Parent) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root\modal-loop.stdout" -RedirectStandardError "$root\modal-loop.stderr"
$p.PriorityClass = 'Idle'
"loop driver pid $($p.Id) $($p.PriorityClass) init=$Init gen=$Gen K=$K"
