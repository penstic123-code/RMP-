# 续跑未完成的 D 组长训（batch=32 防止 N48 + T=512 溢出 8GB 显存）
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$P = "Z:\miniconda\envs\dspn\python.exe"
$Sleep = '{\"dream\":2.0,\"rewire\":false}'

function Run-Full($Tag, $ExtraStr, $Data, $Iters, $T, $Batch, $SmfK) {
    Write-Host "`n==================== [TRAIN] $Tag ($Data, iters=$Iters, T=$T, batch=$Batch) ====================" -ForegroundColor Cyan
    & $P dspn_text.py train dspn --size S --data $Data --iters $Iters --T $T --batch $Batch --tag $Tag --extra $ExtraStr
    Write-Host "==================== [EMERGENCE] $Tag ====================" -ForegroundColor Yellow
    & $P tools/rmp_emergence.py --tag $Tag --extra $ExtraStr --data $Data --rows 32
    $LJson = '{\"freeze_enc\":true,\"smf_k\":' + $SmfK + '}'
    Write-Host "==================== [ONLINE smf_k=$SmfK] $Tag ====================" -ForegroundColor Green
    & $P tools/rmp_online.py --tag $Tag --extra $ExtraStr --modes frozen --sleep $Sleep --name "_smf$SmfK" --learner $LJson
}

$N48k2 = '{\"net\":\"wave\",\"k\":2,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":48,\"init\":\"spread\",\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true}'

Run-Full "_whf_tiny_n48k2_long" $N48k2 "tiny" 12000 512 32 4
Run-Full "_whf_n48k2_long512"   $N48k2 "task4" 9000 512 32 4

Write-Host "`n==================== 10H SLEEP QUEUE FINISHED ====================" -ForegroundColor Magenta
