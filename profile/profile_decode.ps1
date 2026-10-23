# profile_decode.ps1: run colibri on a fixed prompt and sample the engine once a second.
# Writes a CSV of CPU use (as a share of all logical cores), disk bytes read, and memory, plus colibri's own log.
# Usage: .\profile_decode.ps1 -Label baseline [-Tokens 24] [-Ram 16] [-Env @{ DIRECT='0' }] [-Engine C:\path\deepseek_v4.exe]
# -Engine runs another build of the engine through colibri's COLI_ENGINE override; empty means the shipped one.
# Refuses to run unless the laptop is on mains power (Win32_Battery.BatteryStatus = 2); -AllowBattery runs anyway
# (tag such runs as battery runs: absolute speed on battery is about half) but still stops below 30% charge.
# -Summary <csv> appends one row per run (the columns of runs/engine_experiments.csv), using -Experiment and -EngineName.
param(
    [string]$Label = "baseline",
    [int]$Tokens = 24,
    [int]$Ram = 16,
    [hashtable]$Env = @{},
    [string]$Engine = "",
    [string]$Model = "C:\Users\sadan\models\deepseek-v4-flash",
    [string]$Colibri = "C:\Users\sadan\projects\colibri\coli.cmd",
    [string]$Prompt = "Explain in three sentences why a mixture-of-experts model needs less compute per token than a dense model of the same size.",
    [string]$Summary = "",
    [string]$Experiment = "",
    [string]$EngineName = "",
    [switch]$AllowBattery
)
$battery = Get-CimInstance Win32_Battery
if ($battery -and $AllowBattery -and $battery.EstimatedChargeRemaining -lt 30) {
    throw "Battery at $($battery.EstimatedChargeRemaining)%; refusing to run $Label."
}
if ($battery -and $battery.BatteryStatus -ne 2 -and -not $AllowBattery) {
    throw "Not on mains power (BatteryStatus=$($battery.BatteryStatus)); refusing to run $Label."
}
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

if ($Summary) {
    # Decode speed is colibri's own timing: tokens after the first over the time after the first.
    # The output hash is the MD5 of the generated_text line (with its newline), first 8 hex digits.
    $err = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes("$log.err"))
    $inv = [cultureinfo]::InvariantCulture
    $gen = [int]([regex]::Match($err, "v4_tokens .*?generated=(\d+)").Groups[1].Value)
    $hit = [regex]::Match($err, "hit_rate=([\d.]+)").Groups[1].Value
    $tm = [regex]::Match($err, "time_to_first_token=([\d.]+)s after_first=([\d.]+)s")
    $after = [double]::Parse($tm.Groups[2].Value, $inv)
    $line = [regex]::Match($err, "(?m)^generated_text=[^\r\n]*").Value
    $md5 = [Security.Cryptography.MD5]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes("$line`n"))
    $hash = (($md5 | ForEach-Object { $_.ToString("x2") }) -join "").Substring(0, 8)
    $rows = Import-Csv $csv
    $cpuMean = ($rows | ForEach-Object { [double]::Parse($_.cpu_pct_of_machine, $inv) } | Measure-Object -Average).Average
    $rdMean = ($rows | ForEach-Object { [double]::Parse($_.read_GBps, $inv) } | Measure-Object -Average).Average
    $envText = if ($Env.Count) { ($Env.Keys | Sort-Object | ForEach-Object { "$_=$($Env[$_])" }) -join ";" } else { "default" }
    if (-not (Test-Path $Summary)) {
        "run,experiment,engine,env,decode_tok_s,ttft_s,hit_rate_pct,mean_cpu_pct,mean_read_GBps,output_md5_8" | Set-Content $Summary
    }
    [string]::Format($inv, "{0},{1},{2},{3},{4:F3},{5},{6},{7:F1},{8:F2},{9}",
        "$stamp-$Label", $Experiment, $EngineName, $envText, (($gen - 1) / $after),
        $tm.Groups[1].Value, $hit, $cpuMean, $rdMean, $hash) | Add-Content $Summary
}
