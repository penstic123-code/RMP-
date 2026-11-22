#!/bin/bash
# RMP 23.11：分层预测、越深越慢、按任务分工（2026-11-21）。基线 N24 spread（E64、E76、E77）。一项失败不影响后面。
cd "$(dirname "$0")/../.."
P=Z:/miniconda/envs/dspn/python.exe
B='"net":"wave","k":4,"G":4,"D":32,"pd":32,"pm":8,"phidden":64,"N":24,"init":"spread"'
SL='{"dream":2.0,"rewire":false}'
run() {  # tag extra data online?
  echo "== train $1"; $P dspn_text.py train dspn --size S --data $3 --iters 3000 --tag $1 --extra "$2" 2>&1 | grep -a "^(\|Error"
  echo "== emergence $1"; $P tools/rmp_emergence.py --tag $1 --extra "$2" --data $3 --rows 32 2>&1 | grep -a "^{\|Error"
  [ "$4" = on ] && { echo "== online $1"; $P tools/rmp_online.py --tag $1 --extra "$2" --modes frozen,plastic --sleep "$SL" 2>&1 | grep -a "frozen\|plastic\|Error"; }
}
onl() { echo "== online $1"; $P tools/rmp_online.py --tag $1 --extra "$2" --modes frozen,plastic --sleep "$SL" --name "$3" 2>&1 | grep -a "frozen\|plastic\|Error"; }
# 2026-11-21 修多波重置 bug 后：_wpred 的训练与分工指标（故事数据无边界字节）仍有效，只重跑在线；E76、E83、E84 的在线也重跑（对话数据含边界字节）
onl _wpred "{$B,\"pred_w\":0.1}"
onl _wspread_s1 "{$B}" _fix
onl _wctx97 "{$B,\"rule\":\"ctx\",\"ctx_decay\":0.97}" _fix
onl _wctx9 "{$B,\"rule\":\"ctx\",\"ctx_decay\":0.9}" _fix
run _wtau    "{$B,\"tau_prior\":4.0}" tiny on
run _wptc    "{$B,\"pred_w\":0.1,\"tau_prior\":4.0,\"rule\":\"ctx\"}" tiny on
run _wtask   "{$B}" task4
run _wtaskc  "{$B,\"rule\":\"ctx\"}" task4
run _wtaskptc "{$B,\"pred_w\":0.1,\"tau_prior\":4.0,\"rule\":\"ctx\"}" task4
bash tools/queues/rmp_scale_forget.sh
echo "== ALL DONE"
