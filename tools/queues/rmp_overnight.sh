#!/bin/bash
# RMP 通宵队列（2026-11-21）：一项失败不影响后面（用 ; 串）。每项开头打印标记，结果 JSON 在 runs/。
cd "$(dirname "$0")/../.."
P=Z:/miniconda/envs/dspn/python.exe
B='"net":"wave","k":4,"G":4,"D":32,"pd":32,"pm":8,"phidden":64'
S24="{$B,\"N\":24,\"init\":\"spread\"}"
SP="{$B,\"N\":28,\"init\":\"spread\",\"spare\":1}"
SL='{"dream":2.0,"rewire":false}'
tr() { echo "== train $1 $3"; $P dspn_text.py train dspn --size S --data tiny --iters ${4:-3000} --seed ${5:-0} --tag $1 --extra "$2" 2>&1 | grep -a "^(\|完成\|Error\|发散"; }
em() { echo "== emergence $1"; $P tools/rmp_emergence.py --tag $1 --extra "$2" 2>&1 | grep -a "^{\|→\|Error"; }
on() { echo "== online $1 $3 $4"; $P tools/rmp_online.py --tag $1 --extra "$2" --modes $3 --learner "${4:-{\}}" --sleep "$SL" --name "$5" 2>&1 | grep -a "frozen\|plastic\|→\|Error"; }
# 1 新奇分配第三版（新模块输出增益从 0 起、偏置 −0.5）与同检查点对照（睡眠用 E70 的最好设置）
on _wspare "$SP" frozen '{}' _base2
on _wspare "$SP" frozen '{"allocate":1,"novel_ratio":1.5,"spare_bias":-0.5}' _alloc3
# 2 P2b-T1 连接规则：哈希、线性门（学到的地址 = E64 spread）
tr _whash "{$B,\"N\":24,\"init\":\"spread\",\"rule\":\"hash\"}"; em _whash "{$B,\"N\":24,\"init\":\"spread\",\"rule\":\"hash\"}"
tr _wgate "{$B,\"N\":24,\"init\":\"spread\",\"rule\":\"gate\"}"; em _wgate "{$B,\"N\":24,\"init\":\"spread\",\"rule\":\"gate\"}"
# 3 P2b-T2 路由耗时
echo "== router timing"; $P tools/rmp_router_timing.py 2>&1 | grep -a "N\|→\|Error"
# 4 P5 深度生长第三版：最后 20% 不移波
G3='"net":"wave","k":4,"D":32,"pd":32,"pm":8,"phidden":64,"N":24,"init":"flat","grow_every":250,"grow_until":2400'
tr _wgrow3 "{$G3}"; em _wgrow3 "{$G3}"
# 5 补种子
tr _wspread_s1 "$S24" "" 3000 1; em _wspread_s1 "$S24"; on _wspread_s1 "$S24" frozen,plastic '{}' ""
tr _wspread_s2 "$S24" "" 3000 2; em _wspread_s2 "$S24"
# 6 P4 规模初探：每拍每波选 4 个不变，N 增大
for N in 16 48 96 192; do C="{$B,\"N\":$N,\"init\":\"spread\"}"; tr _wN$N "$C"; em _wN$N "$C"; done
# 7 长训练：N24 spread 1.2 万步
tr _wlong "$S24" "" 12000; em _wlong "$S24"; on _wlong "$S24" frozen,plastic '{}' ""
echo "== 全部完成"
