"""Docs checks. Exit 1 on any red item.

- links:    relative links in README, CLAUDE.md and every live docs/**/*.md (archive excluded) point to files that exist (防断链)
- runs:     every runs/*.json is linked from docs/experiments/*.md (防结果漏登记)
- single:   no design doc copies (DSPN*.md) outside docs/archive/ (防出现第二份设计)
- WARN long: a live doc over 12,000 characters (该拆分成章节文件并在目录页登记)
- WARN line: a non-table line over 600 characters (一段写成一行，难扫读；改成要点)
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
red, warn = [], []

pages = [root / 'README.md', root / 'CLAUDE.md',
         *sorted(p for p in (root / 'docs').rglob('*.md') if 'archive' not in p.relative_to(root / 'docs').parts)]
for page in pages:
    text = page.read_text(encoding='utf-8')
    if len(text) > 12000:
        warn.append(f'WARN long: {page.relative_to(root)} 超过 12,000 字符，该拆分')
    for n, line in enumerate(text.splitlines(), 1):
        if len(line) > 600 and not line.startswith('|'):
            warn.append(f'WARN line: {page.relative_to(root)}:{n} 一行 {len(line)} 字，改成要点')
for page in pages:
    for target in re.findall(r'\]\(([^)#\s]+)', page.read_text(encoding='utf-8')):
        if '://' not in target and not (page.parent / target).exists():
            red.append(f'links: {page.relative_to(root)} -> {target}')

experiments = ''.join(p.read_text(encoding='utf-8') for p in (root / 'docs' / 'experiments').rglob('*.md'))
for run in sorted((root / 'runs').glob('*.json')):
    if run.name not in experiments:
        red.append(f'runs: runs/{run.name} 未在 docs/experiments/ 登记')

for stray in root.rglob('DSPN*.md'):
    if (root / 'docs' / 'archive') not in stray.parents:
        red.append(f'single: {stray.relative_to(root)} 是设计文档副本，应并入 docs/design.md 或移入 docs/archive/')

print('\n'.join(red + warn) or 'OK')
raise SystemExit(1 if red else 0)
