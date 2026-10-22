# wait_sweep.ps1: is the clang build faster because of its code or because of how idle threads wait?
# Flips only the OpenMP wait behaviour: make the shipped engine (libgomp) spin, make the clang build (libomp) sleep.
# Each run appends a row to -Summary (default runs\engine_experiments.csv) tagged with -Experiment.
param([int]$Tokens = 24, [int]$Repeats = 2, [string]$Builds = "C:\Users\sadan\projects\colibri-znver5-build",
      [string]$Experiment = "wait", [string]$Summary = (Join-Path $PSScriptRoot "runs\engine_experiments.csv"))
$model = "C:\Users\sadan\models\deepseek-v4-flash"
$usage = Join-Path $model ".coli_usage"
$snap = Join-Path $PSScriptRoot "runs\coli_usage.snapshot"
if (-not (Test-Path $snap)) { Copy-Item $usage $snap }

$v3 = "$Builds\x86-64-v3\deepseek_v4.exe"
$runs = @(
    @{ Label = "shipped-default";  Name = "shipped-gcc-v3"; Engine = "";  Env = @{} },
    @{ Label = "shipped-spin";     Name = "shipped-gcc-v3"; Engine = "";  Env = @{ GOMP_SPINCOUNT = "infinite" } },
    @{ Label = "clang-v3-default"; Name = "clang-v3";       Engine = $v3; Env = @{} },
    @{ Label = "clang-v3-passive"; Name = "clang-v3";       Engine = $v3; Env = @{ OMP_WAIT_POLICY = "passive" } }
)
for ($r = 1; $r -le $Repeats; $r++) {
    foreach ($c in $runs) {
        Copy-Item $snap $usage -Force
        & (Join-Path $PSScriptRoot "profile_decode.ps1") -Label "$($c.Label)-w$r" -Tokens $Tokens -Engine $c.Engine -Env $c.Env `
            -Summary $Summary -Experiment $Experiment -EngineName $c.Name
    }
}
Copy-Item $snap $usage -Force
