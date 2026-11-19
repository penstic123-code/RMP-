"""按顺序跑一串训练/评测命令（GPU 同一时间只跑一个）。每行一条命令，参数用 JSON 列表写，免去 shell 引号问题。

用法：python tools/run_queue.py queue_file.json
queue_file.json：[["train", "dspn", "--extra", "{\"hub_after\": 2}", "--tag", "_hub"], ["evals", "names", "dspn", "--tag", "_hub"], ...]
首项 train → dspn_text.py train …；evals → evals.py …；其它 → tools/<首项>.py …
"""
import json, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
py = sys.executable
for cmd in json.load(open(sys.argv[1], encoding='utf-8-sig')):   # PowerShell 5 写的文件带 BOM
    head, rest = cmd[0], cmd[1:]
    script = {'train': ['dspn_text.py', 'train'], 'evals': ['evals.py']}.get(head, [f'tools/{head}.py'])
    print('>>>', ' '.join(script + rest), flush=True)
    subprocess.run([py, *script, *rest], cwd=root)
