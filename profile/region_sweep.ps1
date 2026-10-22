# region_sweep.ps1: one OpenMP parallel region per layer for the routed experts (layer_region.patch)
# against the unmodified clang builds, for both instruction sets and both OpenMP wait policies.
# All eight conditions interleaved per repeat; same usage snapshot before every run.
param([int]$Tokens = 24, [int]$Repeats = 2, [string]$Builds = "C:\Users\sadan\projects\colibri-znver5-build",
      [string]$Experiment = "region", [string]$Summary = (Join-Path $PSScriptRoot "runs\engine_experiments.csv"))
$model = "C:\Users\sadan\models\deepseek-v4-flash"
$usage = Join-Path $model ".coli_usage"
$snap = Join-Path $PSScriptRoot "runs\coli_usage.snapshot"
if (-not (Test-Path $snap)) { Copy-Item $usage $snap }

$engines = [ordered]@{
    "clang-v3"        = "$Builds\x86-64-v3\deepseek_v4.exe"
    "region-v3"       = "$Builds\layer-region\x86-64-v3\deepseek_v4.exe"
    "clang-znver5"    = "$Builds\znver5\deepseek_v4.exe"
    "region-znver5"   = "$Builds\layer-region\znver5\deepseek_v4.exe"
}
$policies = [ordered]@{ "default" = @{}; "passive" = @{ OMP_WAIT_POLICY = "passive" } }
for ($r = 1; $r -le $Repeats; $r++) {
    foreach ($name in $engines.Keys) {
        foreach ($policy in $policies.Keys) {
            Copy-Item $snap $usage -Force
            & (Join-Path $PSScriptRoot "profile_decode.ps1") -Label "$name-$policy-g$r" -Tokens $Tokens `
                -Engine $engines[$name] -Env $policies[$policy] `
                -Summary $Summary -Experiment $Experiment -EngineName $name
        }
    }
}
Copy-Item $snap $usage -Force
