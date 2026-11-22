# 第一道定生死门槛：task3（留一来源）+ N48k4 + 按 3 个旧来源独立评测 + lora vs pool_mlp 同收益曲线
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$P = "Z:\miniconda\envs\dspn\python.exe"
$env:TRITON_CACHE_DIR = ".pytest_cache\triton"
$Extra = '{\"net\":\"wave\",\"k\":4,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":48,\"init\":\"spread\",\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true}'
$Sleep = '{\"dream\":2.0,\"rewire\":false}'
$Tag = "_t3_n48k4"

Write-Host "`n==================== [TRAIN] $Tag (task3, iters=3000, T=256) ====================" -ForegroundColor Cyan
& $P dspn_text.py train dspn --size S --data task3 --iters 3000 --T 256 --tag $Tag --extra $Extra
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n==================== [EMERGENCE] $Tag ====================" -ForegroundColor Yellow
& $P tools/rmp_emergence.py --tag $Tag --extra $Extra --data task3 --rows 32
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($cfg in @(
    @{ name="_nosmf_lora";   learner='{\"freeze_enc\":true,\"lr\":0.001}' },
    @{ name="_smf4_lora";    learner='{\"freeze_enc\":true,\"smf_k\":4,\"lr\":0.001}' },
    @{ name="_smf4_mlp_1e3"; learner='{\"freeze_enc\":true,\"smf_k\":4,\"params\":\"pool_mlp\",\"lr\":0.001}' },
    @{ name="_smf4_mlp_3e4"; learner='{\"freeze_enc\":true,\"smf_k\":4,\"params\":\"pool_mlp\",\"lr\":0.0003}' }
)) {
    Write-Host "`n==================== [ONLINE $Tag $($cfg.name)] ====================" -ForegroundColor Green
    & $P tools/rmp_online.py --tag $Tag --extra $Extra --modes frozen --old_data task3 --bg_data task3 --seq_data chat3 --seq_bytes 65536 --eval_every 16384 --sleep $Sleep --name $cfg.name --learner $cfg.learner
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n==================== TASK3 GATE FINISHED ====================" -ForegroundColor Magenta
