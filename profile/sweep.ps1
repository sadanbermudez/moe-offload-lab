# sweep.ps1: one profiled run per configuration, each starting from the same learned expert-usage file,
# so colibri's hot-expert pinning (which learns across runs) cannot make later runs look faster.
param([int]$Tokens = 24)
$model = "C:\Users\sadan\models\deepseek-v4-flash"
$usage = Join-Path $model ".coli_usage"
$snap = Join-Path $PSScriptRoot "runs\coli_usage.snapshot"
if (-not (Test-Path $snap)) { Copy-Item $usage $snap }

$configs = [ordered]@{
    "baseline"  = @{}
    "omp24"     = @{ OMP_NUM_THREADS = "24" }
    "omp6"      = @{ OMP_NUM_THREADS = "6" }
    "pipe-off"  = @{ PIPE = "0" }
}
foreach ($name in $configs.Keys) {
    Copy-Item $snap $usage -Force
    & (Join-Path $PSScriptRoot "profile_decode.ps1") -Label $name -Tokens $Tokens -Env $configs[$name]
}
Copy-Item $snap $usage -Force
