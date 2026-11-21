# SMF 甜点扫描 (tiny_n48k2 底座) + tiny_n48k2 3 种子复核
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$P = "Z:\miniconda\envs\dspn\python.exe"
$N48k2 = '{\"net\":\"wave\",\"k\":2,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":48,\"init\":\"spread\",\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true}'
$Sleep = '{\"dream\":2.0,\"rewire\":false}'

# 1. 在 tiny_n48k2 (seed 0) 上扫 SMF 参数与无 SMF 对照（每组 ~4 分钟）
foreach ($cfg in @(
    @{ name="_nosmf_lr1e3";   learner='{\"freeze_enc\":true}' },
    @{ name="_smf8_lr1e3";    learner='{\"freeze_enc\":true,\"smf_k\":8}' },
    @{ name="_smf4_lr3e3";    learner='{\"freeze_enc\":true,\"smf_k\":4,\"lr\":0.003}' },
    @{ name="_smf8_lr3e3";    learner='{\"freeze_enc\":true,\"smf_k\":8,\"lr\":0.003}' }
)) {
    Write-Host "`n==================== [ONLINE tiny_n48k2 $($cfg.name)] ====================" -ForegroundColor Green
    & $P tools/rmp_online.py --tag _whf_tiny_n48k2 --extra $N48k2 --modes frozen --sleep $Sleep --name $cfg.name --learner $cfg.learner
}

# 2. tiny_n48k2 补齐 seed 1, 2（验证严苛顺序任务在 3 种子上复现）
foreach ($s in @(1, 2)) {
    $tag = "_whf_tiny_n48k2_s$s"
    Write-Host "`n==================== [TRAIN] $tag ====================" -ForegroundColor Cyan
    & $P dspn_text.py train dspn --size S --data tiny --iters 3000 --T 256 --seed $s --tag $tag --extra $N48k2
    Write-Host "==================== [EMERGENCE] $tag ====================" -ForegroundColor Yellow
    & $P tools/rmp_emergence.py --tag $tag --extra $N48k2 --data tiny --rows 32
    Write-Host "==================== [ONLINE smf_k=4] $tag ====================" -ForegroundColor Green
    & $P tools/rmp_online.py --tag $tag --extra $N48k2 --modes frozen --sleep $Sleep --name "_smf4" --learner '{\"freeze_enc\":true,\"smf_k\":4}'
}

Write-Host "`n==================== SMF TUNE & SEEDS FINISHED ====================" -ForegroundColor Magenta
