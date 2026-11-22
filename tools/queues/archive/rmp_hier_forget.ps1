# RMP 层级与抗遗忘完整实验队列（Phase 1-3，单种子筛选；约 3-4 小时）
# 用法：powershell -ExecutionPolicy Bypass -File tools/queues/rmp_hier_forget.ps1
$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$P = "Z:\miniconda\envs\dspn\python.exe"
$Base = '\"net\":\"wave\",\"k\":4,\"G\":4,\"D\":32,\"pd\":32,\"pm\":8,\"phidden\":64,\"N\":24,\"init\":\"spread\"'
$Sleep = '{\"dream\":2.0,\"rewire\":false}'

function Run-Exp($Tag, $ExtraFields, $Data, $RunOnline, $LearnerJson) {
    $Extra = if ($ExtraFields) { "{$Base,$ExtraFields}" } else { "{$Base}" }
    Write-Host "`n==================== [TRAIN] $Tag ($Data) ====================" -ForegroundColor Cyan
    & $P dspn_text.py train dspn --size S --data $Data --iters 3000 --tag $Tag --extra $Extra
    Write-Host "==================== [EMERGENCE] $Tag ====================" -ForegroundColor Yellow
    & $P tools/rmp_emergence.py --tag $Tag --extra $Extra --data $Data --rows 32
    if ($RunOnline) {
        Write-Host "==================== [ONLINE] $Tag ====================" -ForegroundColor Green
        $LArg = if ($LearnerJson) { $LearnerJson } else { '{}' }
        & $P tools/rmp_online.py --tag $Tag --extra $Extra --modes frozen,plastic --sleep $Sleep --learner $LArg
    }
}

function Run-OnlineOnly($Tag, $ExtraFields, $NameSuffix, $LearnerJson) {
    $Extra = if ($ExtraFields) { "{$Base,$ExtraFields}" } else { "{$Base}" }
    Write-Host "`n==================== [ONLINE-ONLY] $Tag$NameSuffix ====================" -ForegroundColor Green
    & $P tools/rmp_online.py --tag $Tag --extra $Extra --modes frozen --sleep $Sleep --name $NameSuffix --learner $LearnerJson
}

# ---------------- Phase 1: 层级 2x2 架构压力实验 (TinyStories) ----------------
Run-Exp "_whf_h1" '\"bottleneck\":true' "tiny" $false $null
Run-Exp "_whf_h2" '\"read_last\":true' "tiny" $false $null
Run-Exp "_whf_h3" '\"bottleneck\":true,\"read_last\":true' "tiny" $false $null
Run-Exp "_whf_h4" '\"bottleneck\":true,\"read_last\":true,\"wave_query\":true' "tiny" $true '{\"freeze_enc\":true}'
Run-Exp "_whf_h5" '\"bottleneck\":true,\"read_last\":true,\"wave_query\":true,\"pred_w\":0.1' "tiny" $false $null

# ---------------- Phase 2: task4 多源长程数据的分工与层级 ----------------
Run-Exp "_whf_t1" '\"rule\":\"ctx\",\"ctx_decay\":0.97' "task4" $true '{\"freeze_enc\":true}'
Run-Exp "_whf_t2" '\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true' "task4" $false $null
Run-Exp "_whf_t3" '\"rule\":\"ctx\",\"ctx_decay\":0.97,\"wave_query\":true,\"bottleneck\":true,\"read_last\":true' "task4" $true '{\"freeze_enc\":true}'

# ---------------- Phase 3: 修复漏洞后的抗遗忘与新奇分配重测 ----------------
# F1: 原版底座 + 真正冻结编码层与睡眠路由
Run-Exp "_whf_f1" '' "tiny" $true '{\"freeze_enc\":true}'
# F3/F4: spare=1 底座 + 修好梯度掩码与非零 ogain 的新奇分配
Run-Exp "_whf_spare" '\"spare\":1' "tiny" $false $null
Run-OnlineOnly "_whf_spare" '\"spare\":1' "_alloc_new" '{\"allocate\":1,\"spare_ogain\":0.2,\"freeze_enc\":true}'
Run-OnlineOnly "_whf_spare" '\"spare\":1' "_alloc_split" '{\"allocate\":1,\"alloc_mode\":\"split\",\"freeze_enc\":true}'

Write-Host "`n==================== ALL QUEUE FINISHED ====================" -ForegroundColor Magenta
