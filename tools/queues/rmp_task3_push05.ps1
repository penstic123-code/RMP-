# 在同一 _t3_n48k4 底座上把 chat_drop 推过 0.5，直考第一道门槛：chat_drop>=0.5 时最差旧来源 <=0.05？
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$P = "Z:\miniconda\envs\dspn\python.exe"
$env:TRITON_CACHE_DIR = ".pytest_cache\triton"
$Extra = '{\"net\":\"wave\",\"k\":4,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":48,\"init\":\"spread\",\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true}'
$Sleep = '{\"dream\":2.0,\"rewire\":false}'
$Tag = "_t3_n48k4"

foreach ($cfg in @(
    @{ name="_smf8_mlp_1e3";  bytes=65536;  ev=16384; learner='{\"freeze_enc\":true,\"smf_k\":8,\"params\":\"pool_mlp\",\"lr\":0.001}' },
    @{ name="_smf4_mlp_192k"; bytes=196608; ev=32768; learner='{\"freeze_enc\":true,\"smf_k\":4,\"params\":\"pool_mlp\",\"lr\":0.001}' },
    @{ name="_smf8_mlp_192k"; bytes=196608; ev=32768; learner='{\"freeze_enc\":true,\"smf_k\":8,\"params\":\"pool_mlp\",\"lr\":0.001}' }
)) {
    Write-Host "`n==================== [ONLINE $Tag $($cfg.name)] ====================" -ForegroundColor Green
    & $P tools/rmp_online.py --tag $Tag --extra $Extra --modes frozen --old_data task3 --bg_data task3 --seq_data chat3 --seq_bytes $cfg.bytes --eval_every $cfg.ev --sleep $Sleep --name $cfg.name --learner $cfg.learner
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n==================== TASK3 PUSH-0.5 FINISHED ====================" -ForegroundColor Magenta
