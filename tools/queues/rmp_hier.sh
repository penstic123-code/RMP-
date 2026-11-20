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
run _wpred   "{$B,\"pred_w\":0.1}" tiny on
run _wtau    "{$B,\"tau_prior\":4.0}" tiny on
run _wptc    "{$B,\"pred_w\":0.1,\"tau_prior\":4.0,\"rule\":\"ctx\"}" tiny on
run _wtask   "{$B}" task4
run _wtaskc  "{$B,\"rule\":\"ctx\"}" task4
run _wtaskptc "{$B,\"pred_w\":0.1,\"tau_prior\":4.0,\"rule\":\"ctx\"}" task4
echo "== ALL DONE"
