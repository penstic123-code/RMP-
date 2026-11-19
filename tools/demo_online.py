"""在线学习 + 睡眠巩固演示：教模型生造词 blicket，报告学前 / 在线学后 / 重置后 / 巩固后的 bpc 与旧数据遗忘。

用法：python tools/demo_online.py runs/dspn-S_hub.pt --extra '{"hub_after": 2}' [--out runs/xxx.json]
"""
import argparse, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspn_text as D
from online import demo_new_word

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('--size', default='S'); ap.add_argument('--extra', default='{}')
ap.add_argument('--kind', default='dspn', choices=['dspn'])
ap.add_argument('--learner', default='{}', help='OnlineLearner 参数 JSON，如 {"lr": 3e-4, "params": "lora", "replay": 0.25, "wd": 0.01}')
ap.add_argument('--sleep', default='{}', help='sleep 参数 JSON，如 {"steps": 0} 或 {"anchor": 1.0, "lr": 1e-4}')
ap.add_argument('--day',type=int, default=0, help='教新词前先在线读多少字节普通故事（填充重放池）')
ap.add_argument('--tag', default='')
a = ap.parse_args()
torch.manual_seed(0)
model = D.build(a.kind, a.size, **json.loads(a.extra)).to(D.DEV)
model.load_state_dict(torch.load(a.ckpt, map_location=D.DEV))
res = demo_new_word(model, D.DEV, day_bytes=a.day, sleep=json.loads(a.sleep), **json.loads(a.learner))
for k, v in res.items():
    print(k, v)
out = D.ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_online-{Path(a.ckpt).stem}{a.tag}.json'
json.dump({'ckpt': a.ckpt, 'extra': json.loads(a.extra), **res}, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('→', out.name)
