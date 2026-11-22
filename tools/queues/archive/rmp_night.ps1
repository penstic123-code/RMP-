# RMP 夜间自动实验队列（约 4-5 小时）：SMF 补存、瓶颈消融、高竞争比例、3种子复核、T=512 长训
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$P = "Z:\miniconda\envs\dspn\python.exe"
$Base = '\"net\":\"wave\",\"k\":4,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":24,\"init\":\"spread\"'
$Sleep = '{\"dream\":2.0,\"rewire\":false}'

# 等待上一轮 task-246 (_whf_t3_wide) 完全退出，确保 GPU 单任务运行
Write-Host "Waiting for previous GPU job to finish..." -ForegroundColor DarkGray
while (Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*dspn*" }) {
    Start-Sleep -Seconds 15
}
Write-Host "GPU free, starting night queue." -ForegroundColor Cyan

function Run-Night($Tag, $ExtraStr, $Data, $Iters, $T, $Seed, $SmfK) {
    Write-Host "`n==================== [TRAIN] $Tag ($Data, iters=$Iters, T=$T, seed=$Seed) ====================" -ForegroundColor Cyan
    & $P dspn_text.py train dspn --size S --data $Data --iters $Iters --T $T --seed $Seed --tag $Tag --extra $ExtraStr
    Write-Host "==================== [EMERGENCE] $Tag ====================" -ForegroundColor Yellow
    & $P tools/rmp_emergence.py --tag $Tag --extra $ExtraStr --data $Data --rows 32
    if ($SmfK -ge 0) {
        $LJson = if ($SmfK -gt 0) { '{\"freeze_enc\":true,\"smf_k\":' + $SmfK + '}' } else { '{\"freeze_enc\":true}' }
        Write-Host "==================== [ONLINE smf_k=$SmfK] $Tag ====================" -ForegroundColor Green
        & $P tools/rmp_online.py --tag $Tag --extra $ExtraStr --modes frozen --sleep $Sleep --name "_smf$SmfK" --learner $LJson
    }
}

# 1. 补存 _whf_f1 的 SMF k=4 / k=8 JSON（修复序列化后）
& $P tools/rmp_online.py --tag _whf_f1 --extra "{$Base}" --modes frozen --sleep $Sleep --name "_smf4" --learner '{\"freeze_enc\":true,\"smf_k\":4}'
& $P tools/rmp_online.py --tag _whf_f1 --extra "{$Base}" --modes frozen --sleep $Sleep --name "_smf8" --learner '{\"freeze_enc\":true,\"smf_k\":8}'

# 2. 瓶颈消融：ctx + bottleneck + wave_query（不开 read_last），看是否既保住 bpc 又有逐波层级
$Btn = '{' + $Base + ',\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true}'
Run-Night "_whf_btn" $Btn "task4" 3000 256 0 4

# 3. 高竞争压力：N=48, G=4 (每波 12 个模块)，对比 k=2 (16.7%选中) vs k=4 (33%选中)
$N48k2 = '\"net\":\"wave\",\"k\":2,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":48,\"init\":\"spread\",\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true'
$N48k4 = '\"net\":\"wave\",\"k\":4,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":48,\"init\":\"spread\",\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true'
Run-Night "_whf_n48k2" "{$N48k2}" "task4" 3000 256 0 4
Run-Night "_whf_n48k4" "{$N48k4}" "task4" 3000 256 0 4
# 同配置在纯 TinyStories 上看故事->对话抗遗忘（严苛顺序任务，A不含B）
Run-Night "_whf_tiny_n48k2" "{$N48k2}" "tiny" 3000 256 0 4

# 4. 3 种子复核（_whf_btn 在 task4 与 tiny 上补 seed 1, 2）
Run-Night "_whf_btn_s1" $Btn "task4" 3000 256 1 4
Run-Night "_whf_btn_s2" $Btn "task4" 3000 256 2 4

# 5. 长上下文 (T=512) + 长训练 (9000 步)：验证深波慢时间常数在长上下文下是否拉开
Run-Night "_whf_long512" $Btn "task4" 9000 512 0 4

Write-Host "`n==================== ALL NIGHT QUEUE FINISHED ====================" -ForegroundColor Magenta
