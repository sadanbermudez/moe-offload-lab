# isa_sweep.ps1: the shipped engine against two clang builds of the same source (see build_engines.sh),
# interleaved so drift in the machine's state hits every engine alike. Same usage snapshot before every run.
# Each run appends a row to -Summary (default runs\engine_experiments.csv) tagged with -Experiment.
param([int]$Tokens = 24, [int]$Repeats = 2, [string]$Builds = "C:\Users\sadan\projects\colibri-znver5-build",
      [string]$Experiment = "isa", [string]$Summary = (Join-Path $PSScriptRoot "runs\engine_experiments.csv"))
$model = "C:\Users\sadan\models\deepseek-v4-flash"
$usage = Join-Path $model ".coli_usage"
$snap = Join-Path $PSScriptRoot "runs\coli_usage.snapshot"
if (-not (Test-Path $snap)) { Copy-Item $usage $snap }

$engines = [ordered]@{
    "shipped"      = ""
    "clang-v3"     = "$Builds\x86-64-v3\deepseek_v4.exe"
    "clang-znver5" = "$Builds\znver5\deepseek_v4.exe"
}
$names = @{ "shipped" = "shipped-gcc-v3"; "clang-v3" = "clang-v3"; "clang-znver5" = "clang-znver5" }
for ($r = 1; $r -le $Repeats; $r++) {
    foreach ($name in $engines.Keys) {
        Copy-Item $snap $usage -Force
        & (Join-Path $PSScriptRoot "profile_decode.ps1") -Label "$name-r$r" -Tokens $Tokens -Engine $engines[$name] `
            -Summary $Summary -Experiment $Experiment -EngineName $names[$name]
    }
}
Copy-Item $snap $usage -Force
