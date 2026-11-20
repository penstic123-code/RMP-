#!/bin/bash
# 遗忘是否随参数量下降（2026-11-21，用户问"是不是参数太少"）：E78 的 N16 / N48 检查点跑同一在线协议，和 N24（E76）比。
cd "$(dirname "$0")/../.."
P=Z:/miniconda/envs/dspn/python.exe
B='"net":"wave","k":4,"G":4,"D":32,"pd":32,"pm":8,"phidden":64'
SL='{"dream":2.0,"rewire":false}'
for N in 16 48; do
  echo "== online _wN$N"; $P tools/rmp_online.py --tag _wN$N --extra "{$B,\"N\":$N,\"init\":\"spread\"}" --modes frozen,plastic --sleep "$SL" 2>&1 | grep -a "frozen\|plastic\|Error"
done
echo "== scale done"
