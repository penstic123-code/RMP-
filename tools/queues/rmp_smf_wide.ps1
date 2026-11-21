# SMF 相对背景选模块抗遗忘 + 放宽通道的 task4 层级对照
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$P = "Z:\miniconda\envs\dspn\python.exe"
$Base = '\"net\":\"wave\",\"k\":4,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":24,\"init\":\"spread\"'
$Sleep = '{\"dream\":2.0,\"rewire\":false}'

Write-Host "`n==================== [SMF k=4 on _whf_f1] ====================" -ForegroundColor Green
& $P tools/rmp_online.py --tag _whf_f1 --extra "{$Base}" --modes frozen --sleep $Sleep --name "_smf4" --learner '{\"freeze_enc\":true,\"smf_k\":4}'

Write-Host "`n==================== [SMF k=8 on _whf_f1] ====================" -ForegroundColor Green
& $P tools/rmp_online.py --tag _whf_f1 --extra "{$Base}" --modes frozen --sleep $Sleep --name "_smf8" --learner '{\"freeze_enc\":true,\"smf_k\":8}'

# 放宽瓶颈通道 (pd=64, phidden=96, D=48) 的 t3 层级对照
$Wide = '\"net\":\"wave\",\"k\":4,\"G\":4,\"D\":48,\"pd\":64,\"pm\":8,\"phidden\":96,\"N\":24,\"init\":\"spread\",\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true,\"read_last\":true'
Write-Host "`n==================== [TRAIN] _whf_t3_wide (task4) ====================" -ForegroundColor Cyan
& $P dspn_text.py train dspn --size S --data task4 --iters 3000 --tag _whf_t3_wide --extra "{$Wide}"
Write-Host "==================== [EMERGENCE] _whf_t3_wide ====================" -ForegroundColor Yellow
& $P tools/rmp_emergence.py --tag _whf_t3_wide --extra "{$Wide}" --data task4 --rows 32
Write-Host "==================== [ONLINE SMF k=4] _whf_t3_wide ====================" -ForegroundColor Green
& $P tools/rmp_online.py --tag _whf_t3_wide --extra "{$Wide}" --modes frozen --sleep $Sleep --name "_smf4" --learner '{\"freeze_enc\":true,\"smf_k\":4}'

Write-Host "`n==================== SMF & WIDE QUEUE FINISHED ====================" -ForegroundColor Magenta
