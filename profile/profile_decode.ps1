# profile_decode.ps1: run colibri on a fixed prompt and sample the engine once a second.
# Writes a CSV of CPU use (as a share of all logical cores), disk bytes read, and memory, plus colibri's own log.
# Usage: .\profile_decode.ps1 -Label baseline [-Tokens 24] [-Ram 16] [-Env @{ DIRECT='0' }] [-Engine C:\path\deepseek_v4.exe]
# -Engine runs another build of the engine through colibri's COLI_ENGINE override; empty means the shipped one.
param(
    [string]$Label = "baseline",
    [int]$Tokens = 24,
    [int]$Ram = 16,
    [hashtable]$Env = @{},
    [string]$Engine = "",
    [string]$Model = "C:\Users\sadan\models\deepseek-v4-flash",
    [string]$Colibri = "C:\Users\sadan\projects\colibri\coli.cmd",
    [string]$Prompt = "Explain in three sentences why a mixture-of-experts model needs less compute per token than a dense model of the same size."
)
$out = Join-Path $PSScriptRoot "runs"
New-Item -ItemType Directory -Force $out | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$csv = Join-Path $out "$stamp-$Label.csv"
$log = Join-Path $out "$stamp-$Label.log"
$logical = [Environment]::ProcessorCount
$exe = if ($Engine) { $Engine } else { Join-Path (Split-Path $Colibri) "deepseek_v4.exe" }

if ($Engine) { $env:COLI_ENGINE = $Engine }
foreach ($k in $Env.Keys) { Set-Item "env:$k" $Env[$k] }
$argv = "/c `"`"$Colibri`" run --model `"$Model`" --ram $Ram --gpu none --ngen $Tokens --temp 0 `"$Prompt`"`""
$p = Start-Process cmd -ArgumentList $argv -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden
if ($Engine) { Remove-Item env:COLI_ENGINE }
foreach ($k in $Env.Keys) { Remove-Item "env:$k" }

"t_s,cpu_pct_of_machine,read_GBps,rss_GB,engine" | Set-Content $csv
$t0 = Get-Date; $prev = $null
while (-not $p.HasExited) {
    Start-Sleep -Milliseconds 1000
    # The engine is the .exe that cmd -> python -> engine starts; find it by its full path.
    $e = Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq $exe } | Select-Object -First 1
    if (-not $e) { continue }
    $now = Get-Date
    $cpu = ($e.KernelModeTime + $e.UserModeTime) / 1e7          # 100 ns units -> seconds
    $rd = [double]$e.ReadTransferCount
    if ($prev -and $prev.id -eq $e.ProcessId) {
        $dt = ($now - $prev.t).TotalSeconds
        $cpuPct = 100 * ($cpu - $prev.cpu) / $dt / $logical
        $gbps = ($rd - $prev.rd) / $dt / 1e9
        [string]::Format([cultureinfo]::InvariantCulture, "{0:F1},{1:F1},{2:F2},{3:F1},{4}",
            ($now - $t0).TotalSeconds, $cpuPct, $gbps, ($e.WorkingSetSize / 1GB), $e.Name) | Add-Content $csv
    }
    $prev = @{ id = $e.ProcessId; t = $now; cpu = $cpu; rd = $rd }
}
"exit $($p.ExitCode) after $([int]((Get-Date) - $t0).TotalSeconds) s -> $csv"
