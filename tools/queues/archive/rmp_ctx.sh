#!/bin/bash
# RMP 按语境路由（rule='ctx'，2026-11-21）：查询取残差流慢滑动平均；与 E76/E77 的 N24 spread 同配置，比分工与在线遗忘。
cd "$(dirname "$0")/../.."
P=Z:/miniconda/envs/dspn/python.exe
B='"net":"wave","k":4,"G":4,"D":32,"pd":32,"pm":8,"phidden":64,"N":24,"init":"spread","rule":"ctx"'
SL='{"dream":2.0,"rewire":false}'
for a in 0.97 0.9; do
  C="{$B,\"ctx_decay\":$a}"; t=_wctx${a#0.}
  echo "== train $t"; $P dspn_text.py train dspn --size S --data tiny --iters 3000 --tag $t --extra "$C" 2>&1 | grep -a "^(\|Error"
  echo "== emergence $t"; $P tools/rmp_emergence.py --tag $t --extra "$C" 2>&1 | grep -a "^{\|Error"
  echo "== online $t"; $P tools/rmp_online.py --tag $t --extra "$C" --modes frozen,plastic --sleep "$SL" 2>&1 | grep -a "frozen\|plastic\|Error"
done
echo "== ALL DONE"
