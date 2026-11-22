# 共享专家消融：shared=1 + routed k=3（与 E106/E107 纯路由 k=4 严格对齐每拍 4 个激活模块）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$P = "Z:\miniconda\envs\dspn\python.exe"
$env:TRITON_CACHE_DIR = ".pytest_cache\triton"
$Extra = '{\"net\":\"wave\",\"k\":3,\"shared\":1,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":48,\"init\":\"spread\",\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true}'
$Sleep = '{\"dream\":2.0,\"rewire\":false}'
$Tag = "_t3_n48_sh1k3"

Write-Host "`n==================== [TRAIN] $Tag (task3, shared=1+k=3, iters=3000) ====================" -ForegroundColor Cyan
& $P dspn_text.py train dspn --size S --data task3 --iters 3000 --T 256 --tag $Tag --extra $Extra
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n==================== [EMERGENCE] $Tag ====================" -ForegroundColor Yellow
& $P tools/rmp_emergence.py --tag $Tag --extra $Extra --data task3 --rows 32
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($cfg in @(
    @{ name="_smf8_mlp_64k";  bytes=65536;  ev=16384; learner='{\"freeze_enc\":true,\"smf_k\":8,\"params\":\"pool_mlp\",\"lr\":0.001}' },
    @{ name="_smf8_mlp_192k"; bytes=196608; ev=32768; learner='{\"freeze_enc\":true,\"smf_k\":8,\"params\":\"pool_mlp\",\"lr\":0.001}' }
)) {
    Write-Host "`n==================== [ONLINE $Tag $($cfg.name)] ====================" -ForegroundColor Green
    & $P tools/rmp_online.py --tag $Tag --extra $Extra --modes frozen --old_data task3 --bg_data task3 --seq_data chat3 --seq_bytes $cfg.bytes --eval_every $cfg.ev --sleep $Sleep --name $cfg.name --learner $cfg.learner
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n==================== TASK3 SHARED=1 FINISHED ====================" -ForegroundColor Magenta
