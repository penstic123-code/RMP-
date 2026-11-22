"""本地网页聊天与长期记忆控制台（零额外依赖，内置 Python HTTP 服务）。

启动方式（PowerShell）：
  $env:TRITON_CACHE_DIR=".pytest_cache\\triton"; $env:HF_HOME=".pytest_cache\\hf"
  Z:\\miniconda\\envs\\dspn\\python.exe tools\\web_chat_sidecar.py --preset large --port 7860

浏览器打开：http://127.0.0.1:7860

特色：
  - 左侧聊天区：支持实时开关「海马体/SQLite 检索」与「RMP 旁路权重」，直观对比“纯原版模型 vs SQLite 检索 vs RMP 权重记忆”；
  - 右侧记忆台：实时增删改 SQLite 长期事实（改完秒级重建索引），添加教学规则并一键触发「睡眠巩固（/sleep）」。
"""
import argparse, json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get('TRITON_CACHE_DIR'):
    os.environ['TRITON_CACHE_DIR'] = str(ROOT / '.pytest_cache' / 'triton')
if not os.environ.get('HF_HOME') and (ROOT / '.pytest_cache' / 'hf').exists():
    os.environ['HF_HOME'] = str(ROOT / '.pytest_cache' / 'hf')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from llm_sidecar import LLMSidecarWrapper, SidecarLearner, SIDECAR_PRESETS, _lex_tokens
from memory_store import MemoryStore
from tools.chat_llm_sidecar import BG_SEED, BASE_SYS, encode_utterance, stage_confirmed_qa, rebuild_hippo_from_db

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>RMP 持续学习助手 · 本地记忆控制台</title>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --card: #273449; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; --ok: #34d399;
    --warn: #fbbf24; --danger: #f87171;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
         background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
  header { padding: 10px 18px; background: var(--panel); border-bottom: 1px solid var(--border);
           display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; font-size: 12px; }
  .badge { background: var(--card); border: 1px solid var(--border); padding: 4px 9px; border-radius: 999px; color: var(--muted); }
  .badge b { color: var(--text); }
  .toggles { display: flex; gap: 12px; align-items: center; font-size: 13px; }
  .toggle { display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none;
            background: var(--card); padding: 5px 10px; border-radius: 8px; border: 1px solid var(--border); }
  .toggle input { accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }
  main { flex: 1; display: grid; grid-template-columns: 1.35fr 1fr; min-height: 0; }
  @media (max-width: 960px) { main { grid-template-columns: 1fr; } }
  .chat-col { display: flex; flex-direction: column; border-right: 1px solid var(--border); min-height: 0; }
  .messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; font-size: 14px; }
  .msg.user { align-self: flex-end; background: #0284c7; color: #fff; border-bottom-right-radius: 4px; }
  .msg.ai { align-self: flex-start; background: var(--panel); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
  .msg.sys { align-self: center; background: transparent; color: var(--muted); font-size: 12px; padding: 4px 8px; }
  .meta { margin-top: 6px; padding-top: 6px; border-top: 1px dashed var(--border); font-size: 11px; color: var(--muted); }
  .meta .pill { display: inline-block; background: var(--card); padding: 1px 6px; border-radius: 4px; margin-right: 5px; }
  .meta .pill.on { color: var(--ok); }
  .meta .pill.off { color: var(--warn); }
  .composer { padding: 12px 16px; background: var(--panel); border-top: 1px solid var(--border); display: flex; gap: 8px; }
  .composer textarea { flex: 1; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 8px;
                       padding: 10px 12px; font-size: 14px; resize: none; height: 46px; font-family: inherit; }
  .composer textarea:focus { outline: none; border-color: var(--accent); }
  button { background: var(--accent); color: #0f172a; border: none; border-radius: 8px; padding: 8px 14px;
           font-weight: 600; font-size: 13px; cursor: pointer; transition: opacity .15s; }
  button:hover { opacity: .88; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  button.sec { background: var(--card); color: var(--text); border: 1px solid var(--border); }
  button.ok { background: var(--ok); color: #0f172a; }
  button.danger { background: transparent; color: var(--danger); border: 1px solid #7f1d1d; padding: 4px 8px; font-size: 12px; }
  button.sm { padding: 4px 8px; font-size: 12px; }

  .side-col { display: flex; flex-direction: column; min-height: 0; background: #131c31; }
  .tabs { display: flex; border-bottom: 1px solid var(--border); background: var(--panel); }
  .tab { flex: 1; text-align: center; padding: 11px 8px; cursor: pointer; font-size: 13px; font-weight: 600;
         color: var(--muted); border-bottom: 2px solid transparent; }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .pane { flex: 1; overflow-y: auto; padding: 14px; display: none; flex-direction: column; gap: 12px; }
  .pane.active { display: flex; }
  .box { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
  .box h3 { margin: 0 0 8px 0; font-size: 13px; color: var(--muted); font-weight: 600; }
  .row { display: flex; gap: 6px; margin-bottom: 6px; }
  input[type=text], input[type=number] {
    background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px;
    padding: 7px 10px; font-size: 13px; width: 100%; font-family: inherit;
  }
  input:focus { outline: none; border-color: var(--accent); }
  .item { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 9px 10px;
          display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 7px; font-size: 13px; }
  .item-body { flex: 1; word-break: break-word; }
  .key-tag { display: inline-block; background: #0369a1; color: #e0f2fe; font-size: 11px; padding: 1px 6px; border-radius: 4px; margin-right: 5px; }
  .st-tag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px; margin-right: 5px; }
  .st-tag.confirmed { background: #065f46; color: #a7f3d0; }
  .st-tag.staged { background: #78350f; color: #fde68a; }
  .hint { font-size: 12px; color: var(--muted); line-height: 1.5; }
  .sleep-log { margin-top: 8px; font-size: 12px; padding: 8px 10px; border-radius: 6px; background: var(--bg);
               border: 1px solid var(--border); white-space: pre-wrap; display: none; }
</style>
</head>
<body>
<header>
  <h1>🧠 RMP 持续学习助手 <span class="badge" id="modelBadge">加载中...</span></h1>
  <div class="toggles">
    <label class="toggle" title="关闭后不从 SQLite/海马体注入任何检索文本，纯靠模型权重回答">
      <input type="checkbox" id="useHippo" checked/> 海马体/SQLite 检索
    </label>
    <label class="toggle" title="关闭后完全旁路 RMP，退回原版冻结 Qwen 模型">
      <input type="checkbox" id="useRmp" checked/> RMP 旁路权重
    </label>
    <label class="toggle" title="开启后每次聊天自动由 AI 判断是否值得学习、跳过闲聊与重复，并自动加入待巩固队列">
      <input type="checkbox" id="autoMem" checked/> 🤖 自动抽取记忆
    </label>
    <button class="sec" onclick="newSession()">新会话</button>
  </div>
  <div class="badges" id="statBadges"></div>
</header>

<main>
  <section class="chat-col">
    <div class="messages" id="messages">
      <div class="msg sys">提示：你可以把右上角「海马体/SQLite 检索」关掉，亲自验证教过的规则是不是真的长进了 RMP 神经网络权重里。</div>
    </div>
    <div class="composer">
      <textarea id="chatInput" placeholder="输入消息（Enter 发送，Shift+Enter 换行）..."></textarea>
      <button id="sendBtn" onclick="sendChat()">发送</button>
    </div>
  </section>

  <aside class="side-col">
    <div class="tabs">
      <div class="tab active" onclick="switchTab(0)">📚 长期事实 (SQLite / 海马体)</div>
      <div class="tab" onclick="switchTab(1)">⚡ 教学规则 &amp; 睡眠 (RMP 权重)</div>
    </div>

    <div class="pane active" id="pane0">
      <div class="box">
        <h3>新增 / 改口覆盖事实（同 key 自动覆盖旧值）</h3>
        <div class="row">
          <input type="text" id="factKey" placeholder="可选 key（如 city / cat / coffee）" style="max-width:170px"/>
          <input type="text" id="factText" placeholder="事实内容（如 I live in Sydney.）"/>
        </div>
        <button onclick="addFact()">写入 SQLite + 海马体</button>
      </div>
      <div class="box" style="flex:1">
        <h3>当前活跃事实（点击编辑或彻底删除）</h3>
        <div id="factList" class="hint">加载中...</div>
      </div>
    </div>

    <div class="pane" id="pane1">
      <div class="box">
        <h3>教一条新规则 / 暗号 / 习惯（将巩固进 RMP 权重）</h3>
        <div class="row"><input type="text" id="ruleQ" placeholder="触发问题（如 What is project Aurora code?）"/></div>
        <div class="row"><input type="text" id="ruleA" placeholder="期望回答（如 The code is ORION-94.）"/></div>
        <button onclick="addRule()">加入待巩固队列</button>
      </div>
      <div class="box">
        <h3>🌙 睡眠巩固（把待巩固规则写进 RMP 稀疏模块）</h3>
        <div class="hint" style="margin-bottom:8px">
          只训练冷门 RMP 模块 + 历史规则回放 + 纯冻结原脑 KL 锚定；若旧能力退化超红线会自动回滚。
        </div>
        <div class="row" style="align-items:center">
          <span class="hint" style="white-space:nowrap">睡眠步数：</span>
          <input type="number" id="sleepSteps" value="40" min="10" max="200" style="max-width:90px"/>
          <button class="ok" id="sleepBtn" onclick="runSleep()">开始睡眠巩固</button>
        </div>
        <div class="sleep-log" id="sleepLog"></div>
      </div>
      <div class="box" style="flex:1">
        <h3>规则列表（staged=待巩固 | confirmed=已长进权重）</h3>
        <div id="ruleList" class="hint">加载中...</div>
      </div>
    </div>
  </aside>
</main>

<script>
const $ = id => document.getElementById(id);

function switchTab(i) {
  document.querySelectorAll('.tab').forEach((el, idx) => el.classList.toggle('active', idx === i));
  document.querySelectorAll('.pane').forEach((el, idx) => el.classList.toggle('active', idx === i));
}

async function api(path, body) {
  const opt = body !== undefined ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  return await r.json();
}

function addMsg(role, text, meta) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    const hPill = `<span class="pill ${meta.use_hippo?'on':'off'}">SQLite/海马体: ${meta.use_hippo?'ON':'OFF'}</span>`;
    const rPill = `<span class="pill ${meta.use_rmp?'on':'off'}">RMP权重: ${meta.use_rmp?'ON':'OFF'}</span>`;
    const tPill = `<span class="pill">${meta.elapsed_ms} ms</span>`;
    let rec = '';
    if (meta.recalled && meta.recalled.length) {
      rec = '<div style="margin-top:4px">召回事实：' + meta.recalled.map(x => `「${x}」`).join(' ') + '</div>';
    } else if (meta.use_hippo) {
      rec = '<div style="margin-top:4px">召回事实：无</div>';
    }
    m.innerHTML = hPill + rPill + tPill + rec;
    d.appendChild(m);
  }
  $('messages').appendChild(d);
  $('messages').scrollTop = $('messages').scrollHeight;
}

async function refreshState() {
  const s = await api('/api/state');
  $('modelBadge').textContent = `${s.preset} · ${s.model}`;
  $('statBadges').innerHTML =
    `<span class="badge">事实 <b>${s.facts.length}</b></span>` +
    `<span class="badge">已巩固规则 <b>${s.confirmed_count}</b></span>` +
    `<span class="badge">待巩固 <b>${s.staged_count}</b></span>` +
    `<span class="badge">VRAM <b>${s.vram_mb} MB</b></span>` +
    `<span class="badge">DB <b>${s.db_kb} KB</b></span>`;
  // 渲染事实
  if (!s.facts.length) {
    $('factList').innerHTML = '<div class="hint">暂无活跃事实。聊天时或上方表单都可以写入。</div>';
  } else {
    $('factList').innerHTML = s.facts.map(f => `
      <div class="item">
        <div class="item-body">
          ${f.fact_key ? `<span class="key-tag">${esc(f.fact_key)}</span>` : ''}
          <span>#${f.id} ${esc(f.text)}</span>
        </div>
        <div style="display:flex;gap:4px">
          <button class="sec sm" onclick="editFact(${f.id}, ${JSON.stringify(f.text)})">改</button>
          <button class="danger" onclick="delFact(${f.id})">删</button>
        </div>
      </div>`).join('');
  }
  // 渲染规则
  if (!s.rules.length) {
    $('ruleList').innerHTML = '<div class="hint">暂无教学规则。在上方添加后点击「开始睡眠巩固」。</div>';
  } else {
    $('ruleList').innerHTML = s.rules.map(r => `
      <div class="item">
        <div class="item-body">
          <span class="st-tag ${r.status}">${r.status}</span>
          <b>Q:</b> ${esc(r.question)}<br/><b>A:</b> ${esc(r.answer)}
        </div>
        <button class="danger" onclick="delRule(${r.id})">删</button>
      </div>`).join('');
  }
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function sendChat() {
  const txt = $('chatInput').value.trim();
  if (!txt) return;
  $('chatInput').value = '';
  $('sendBtn').disabled = true;
  addMsg('user', txt);
  const d = document.createElement('div');
  d.className = 'msg ai';
  const bodySpan = document.createElement('span');
  bodySpan.textContent = '…';
  d.appendChild(bodySpan);
  $('messages').appendChild(d);
  $('messages').scrollTop = $('messages').scrollHeight;
  try {
    const resp = await fetch('/api/chat_stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: txt,
        use_hippo: $('useHippo').checked,
        use_rmp: $('useRmp').checked,
        auto_mem: $('autoMem').checked,
      }),
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '', started = false;
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.delta) {
          if (!started) { bodySpan.textContent = ''; started = true; }
          bodySpan.textContent += ev.delta;
          $('messages').scrollTop = $('messages').scrollHeight;
        } else if (ev.done) {
          if (!started) bodySpan.textContent = ev.reply || '';
          const m = document.createElement('div');
          m.className = 'meta';
          const hPill = `<span class="pill ${ev.use_hippo?'on':'off'}">SQLite/海马体: ${ev.use_hippo?'ON':'OFF'}</span>`;
          const rPill = `<span class="pill ${ev.use_rmp?'on':'off'}">RMP权重: ${ev.use_rmp?'ON':'OFF'}</span>`;
          const tPill = `<span class="pill">${ev.elapsed_ms} ms</span>`;
          let rec = '';
          if (ev.recalled && ev.recalled.length) {
            rec = '<div style="margin-top:4px">召回事实：' + ev.recalled.map(x => `「${esc(x)}」`).join(' ') + '</div>';
          } else if (ev.use_hippo) {
            rec = '<div style="margin-top:4px">召回事实：无</div>';
          }
          m.innerHTML = hPill + rPill + tPill + rec;
          d.appendChild(m);
          $('messages').scrollTop = $('messages').scrollHeight;
        } else if (ev.auto_mem) {
          const am = ev.auto_mem;
          const box = document.createElement('div');
          box.className = 'meta';
          if (am.action === 'learn') {
            box.innerHTML = `<span class="pill on">🧠 自动加入待巩固</span> ${esc(am.rule_q)} → <b>${esc(am.rule_a)}</b>` +
              (am.updated ? ' <span class="pill off">已更新旧规则</span>' : '') +
              (am.reason ? `<div style="margin-top:2px">理由：${esc(am.reason)}</div>` : '');
          } else if (am.action === 'duplicate') {
            box.innerHTML = `<span class="pill off">♻️ 重复内容已跳过</span> ${esc(am.reason || '已有相同记忆')}`;
          } else {
            box.innerHTML = `<span class="pill">⏭️ 无需学习</span> ${esc(am.reason || '普通闲聊或提问')}`;
          }
          d.appendChild(box);
          $('messages').scrollTop = $('messages').scrollHeight;
        }
      }
    }
    refreshState();
  } catch (e) {
    bodySpan.textContent = '请求失败：' + e;
  } finally {
    $('sendBtn').disabled = false;
    $('chatInput').focus();
  }
}

async function newSession() {
  const res = await api('/api/new_session', {});
  $('messages').innerHTML = `<div class="msg sys">已开启新会话 #${res.session}（对话清空，SQLite 与 RMP 权重记忆保留）</div>`;
  refreshState();
}

async function addFact() {
  const text = $('factText').value.trim();
  const key = $('factKey').value.trim();
  if (!text) return;
  await api('/api/fact/add', {fact_key: key, text});
  $('factText').value = ''; $('factKey').value = '';
  refreshState();
}

async function editFact(id, oldText) {
  const nt = prompt('修改事实 #' + id + '：', oldText);
  if (nt === null || !nt.trim()) return;
  await api('/api/fact/edit', {id, text: nt.trim()});
  refreshState();
}

async function delFact(id) {
  if (!confirm('确认彻底删除事实 #' + id + '？')) return;
  await api('/api/fact/delete', {id});
  refreshState();
}

async function addRule() {
  const q = $('ruleQ').value.trim(), a = $('ruleA').value.trim();
  if (!q || !a) return;
  await api('/api/rule/add', {question: q, answer: a});
  $('ruleQ').value = ''; $('ruleA').value = '';
  refreshState();
}

async function delRule(id) {
  if (!confirm('确认删除规则 #' + id + '？')) return;
  await api('/api/rule/delete', {id});
  refreshState();
}

async function runSleep() {
  $('sleepBtn').disabled = true;
  $('sleepLog').style.display = 'block';
  $('sleepLog').textContent = '🌙 正在睡眠巩固（SMF 稀疏更新 + 历史回放 + 冻结原脑锚定）...';
  try {
    const steps = parseInt($('sleepSteps').value || '40', 10);
    const res = await api('/api/sleep', {steps});
    if (res.steps === 0) {
      $('sleepLog').textContent = 'ℹ️ 待巩固队列为空：请先聊点值得记住的内容（开启自动抽取），或在上方手动添加规则。';
    } else if (res.committed) {
      $('sleepLog').textContent =
        `✅ 睡眠巩固成功！\n` +
        `• 训练步数：${res.steps} 步 | 仅更新了 ${res.consolidated_modules} 个冷门 RMP 模块\n` +
        `• 新规则 Loss：${res.new_loss[0]} → ${res.new_loss[1]}（已学会）\n` +
        `• 通用旧能力波动：${res.anchor_rise >= 0 ? '+' : ''}${res.anchor_rise} | 历史规则波动：${res.hist_rise >= 0 ? '+' : ''}${res.hist_rise}`;
    } else {
      $('sleepLog').textContent =
        `⚠️ 触发安全保护，已自动回滚本次睡眠！\n` +
        `• 新规则 Loss：${res.new_loss[0]} → ${res.new_loss[1]}\n` +
        `• 旧能力波动：${res.anchor_rise} | 历史规则波动：${res.hist_rise}（超过安全阈值）`;
    }
    refreshState();
  } catch (e) {
    $('sleepLog').textContent = '睡眠出错：' + e;
  } finally {
    $('sleepBtn').disabled = false;
  }
}

$('chatInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
refreshState();
</script>
</body>
</html>
"""


class SidecarService:
    def __init__(self, preset='large', model='', db_path='', mem_path='', lr=1e-3, smf_k=8, max_new=120):
        self.lock = threading.Lock()
        self.preset = preset
        cfg = SIDECAR_PRESETS[preset]
        self.model_name = model or cfg['model']
        self.db_path = Path(db_path) if db_path else (ROOT / 'runs/memory.db')
        self.mem_path = Path(mem_path) if mem_path else (ROOT / f'runs/sidecar_{preset}_memory.pt')
        self.max_new = max_new
        print(f'正在加载 [{preset}] {self.model_name} ({DEV})...', flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        lm = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.bfloat16 if DEV == 'cuda' else torch.float32
        ).to(DEV)
        self.w = LLMSidecarWrapper(lm, attach_layer=-4, sidecar_cfg=cfg['sidecar'], hippo_cfg=cfg['hippo'])
        self.w.sidecar.to(DEV); self.w.hippo.to(DEV)
        self.L = SidecarLearner(self.w, lr=lr, smf_k=smf_k)
        self.bg_batches = [self.tok(t, return_tensors='pt').input_ids.to(DEV) for t in BG_SEED]
        self.L.profile_bg(self.bg_batches)
        self.store = MemoryStore(self.db_path)
        prev_model = self.store.get_meta('model_name')
        model_changed = bool(prev_model and prev_model != self.model_name)
        if not model_changed and self.mem_path.exists():
            self.w.load_memory(self.mem_path, map_location=DEV, learner=self.L)
        rebuild_hippo_from_db(self.store, self.w, self.tok)
        self.L.staged.clear(); self.L.confirmed_qa.clear()
        for _, q, ans, st in self.store.list_rules():
            stage_confirmed_qa(self.L, self.tok, q, ans, to_confirmed=(st == 'confirmed' and not model_changed))
        self.store.set_meta('model_name', self.model_name)
        self.history = []

    def state_dict_json(self):
        with self.lock:
            facts = [{'id': r[0], 'fact_key': r[1], 'text': r[2], 'session_id': r[3]} for r in self.store.list_facts()]
            rules = [{'id': r[0], 'question': r[1], 'answer': r[2], 'status': r[3]} for r in self.store.list_rules()]
            vram = round(torch.cuda.memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0
            db_kb = round(self.db_path.stat().st_size / 1024.0, 1) if self.db_path.exists() else 0.0
            return {
                'preset': self.preset, 'model': self.model_name,
                'facts': facts, 'rules': rules,
                'staged_count': len(self.L.staged), 'confirmed_count': len(self.L.confirmed_qa),
                'vram_mb': vram, 'db_kb': db_kb,
            }

    def chat(self, message, use_hippo=True, use_rmp=True):
        with self.lock:
            t0 = time.time()
            recalled = []
            if use_hippo:
                h_u = encode_utterance(self.w, self.tok, message)
                recalled = self.w.hippo.recall_facts(h_u, query_text=message, top_k=3)
            sys_txt = BASE_SYS
            if recalled:
                sys_txt += "\n\nKnown facts about the user (authoritative, from past sessions):\n" + "\n".join(f"- {f}" for f in recalled)
            msgs = [{'role': 'system', 'content': sys_txt}] + self.history[-6:] + [{'role': 'user', 'content': message}]
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = self.tok(prompt, return_tensors='pt').input_ids.to(DEV)
            self.w.eval(); self.w.sidecar.reset_session()
            self.w.bypass = not use_rmp
            try:
                with torch.no_grad():
                    out = self.w.backbone.generate(
                        ids, max_new_tokens=self.max_new, do_sample=False, pad_token_id=self.tok.eos_token_id
                    )
            finally:
                self.w.bypass = False
            reply = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            self.history.extend([{'role': 'user', 'content': message}, {'role': 'assistant', 'content': reply}])
            return {
                'reply': reply, 'recalled': recalled, 'use_hippo': use_hippo, 'use_rmp': use_rmp,
                'elapsed_ms': int((time.time() - t0) * 1000),
            }

    def _auto_judge_memory(self, message):
        """用冻结原脑（bypass=True）自动判断：该学 / 跳过闲聊 / 跳过重复，并加问句拦截与防幻觉原文校验。"""
        t = message.strip()
        if len(t) < 4:
            return {'action': 'ignore', 'reason': '内容过短'}
        tl = t.lower()
        q_cues = ('吗', '呢', '什么', '哪', '谁', '怎么', '为什么', '记得', '知道', '多少', '几',
                  'what ', 'who ', 'where ', 'when ', 'why ', 'how ', 'do you ')
        if t.endswith(('?', '？')) or any(q in tl for q in q_cues):
            return {'action': 'ignore', 'reason': '检测到提问句，不写入新记忆'}

        existing_rules = self.store.list_rules()
        mem_lines = [f"- {q} => {a}" for _, q, a, _ in existing_rules[-20:]]
        mem_ctx = "\n".join(mem_lines) if mem_lines else "（暂无）"
        sys_p = (
            '你是持续学习系统的记忆门控器。判断用户这句话是否明确陈述了值得长期记住的新事实'
            '（如个人资料、宠物、喜好、工作、项目暗号、改口更新）。\n'
            '严禁编造用户原话里没提到的任何信息！提问句一律返回 ignore。\n'
            '只输出一行合法 JSON，不要任何多余文字：\n'
            '1) 闲聊/打招呼/提问/无明确新事实：{"action":"ignore","reason":"简短中文理由"}\n'
            '2) 与已有记忆含义重复：{"action":"duplicate","reason":"简短中文理由"}\n'
            '3) 值得长期记住：{"action":"learn","fact_key":"英文短键如dog_name或空","fact_text":"简短事实句","rule_q":"以用户口吻提问(我...)","rule_a":"以助手口吻回答(你...)","reason":"简短中文理由"}\n'
            '示例1：输入「你好，我有一条dog，叫旺财」→ '
            '{"action":"learn","fact_key":"dog_name","fact_text":"用户的狗叫旺财。","rule_q":"我的狗叫什么名字？","rule_a":"你的狗叫旺财。","reason":"包含用户宠物名称"}\n'
            '示例2：输入「我有什么爱好，你记得吗」→ {"action":"ignore","reason":"用户在提问，未提供新事实"}'
        )
        usr_p = f"已有记忆：\n{mem_ctx}\n\n用户新消息：{t}"
        prompt = self.tok.apply_chat_template(
            [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': usr_p}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = self.tok(prompt, return_tensors='pt').input_ids.to(DEV)
        self.w.eval(); self.w.bypass = True
        try:
            with torch.no_grad():
                out = self.w.backbone.generate(ids, max_new_tokens=160, do_sample=False, pad_token_id=self.tok.eos_token_id)
        finally:
            self.w.bypass = False
        raw = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        m = re.search(r'\{.*\}', raw, re.S)
        if not m:
            return {'action': 'ignore', 'reason': '未检测到明确可记忆事实'}
        try:
            res = json.loads(m.group(0))
        except Exception:
            return {'action': 'ignore', 'reason': '未检测到明确可记忆事实'}

        action = str(res.get('action', 'ignore')).lower()
        reason = str(res.get('reason', '')).strip()
        if action != 'learn':
            return {'action': 'duplicate' if action == 'duplicate' else 'ignore', 'reason': reason}

        rq = str(res.get('rule_q', '')).strip()
        ra = str(res.get('rule_a', '')).strip()
        fk = str(res.get('fact_key', '')).strip().lower()
        ft = str(res.get('fact_text', '')).strip()
        if not rq or not ra:
            return {'action': 'ignore', 'reason': '抽取结果不完整，已忽略'}

        # 防幻觉原文核验：答案中新出现的关键词（去掉问题里的词和虚词）必须能在用户原话里找到
        stop_zh = set('你的我的用户的他她它是叫有在喜欢爱喝吃住名字爱好东西问题回答了呢吗啊吧呀着过')
        q_toks = _lex_tokens(rq)
        a_new = [w for w in _lex_tokens(ra) if w not in q_toks and not (len(w) == 1 and w in stop_zh)]
        if a_new:
            hit = sum(1 for w in a_new if w in tl)
            if hit / len(a_new) < 0.4:
                return {'action': 'ignore', 'reason': '拦截 AI 臆测：抽取内容未在你的原话中出现'}

        # 代码级二次去重 + 改口覆盖兜底
        q_lex, a_lex = _lex_tokens(rq), _lex_tokens(ra)
        updated = False
        for rid, oq, oa, _ in existing_rules:
            oq_lex, oa_lex = _lex_tokens(oq), _lex_tokens(oa)
            qj = len(q_lex & oq_lex) / max(len(q_lex | oq_lex), 1)
            aj = len(a_lex & oa_lex) / max(len(a_lex | oa_lex), 1)
            if (rq == oq and ra == oa) or (qj >= 0.7 and aj >= 0.7):
                return {'action': 'duplicate', 'reason': f'与已有规则「{oq}」重复，已跳过'}
            if qj >= 0.65 and aj < 0.7:
                self.store.delete_rule(rid)
                updated = True

        self.store.add_rule(rq, ra, status='staged')
        if updated:
            self.L.staged.clear(); self.L.confirmed_qa.clear()
            for _, q_, a_, st_ in self.store.list_rules():
                stage_confirmed_qa(self.L, self.tok, q_, a_, to_confirmed=(st_ == 'confirmed'))
        else:
            stage_confirmed_qa(self.L, self.tok, rq, ra, to_confirmed=False)

        if ft:
            h_u = encode_utterance(self.w, self.tok, ft)
            before = {self.w.hippo.slot_texts[s] for s in (self.w.hippo.used & (~self.w.hippo.superseded)).nonzero(as_tuple=False)[:, 0].tolist()}
            self.w.hippo.write_fact(h_u, ft, score=2.0, fact_key=fk)
            after = {self.w.hippo.slot_texts[s] for s in (self.w.hippo.used & (~self.w.hippo.superseded)).nonzero(as_tuple=False)[:, 0].tolist()}
            self.store.add_fact(ft, fact_key=fk, session_id=self.w.hippo.cur_session, supersede_texts=before - after)
        self.w.save_memory(self.mem_path, learner=self.L)
        return {'action': 'learn', 'rule_q': rq, 'rule_a': ra, 'fact_key': fk, 'updated': updated, 'reason': reason}

    def chat_stream(self, message, use_hippo=True, use_rmp=True, auto_mem=True):
        """逐字流式生成，首字 ~0.3 秒即出；生成完毕后自动判断是否值得加入待巩固队列。"""
        with self.lock:
            t0 = time.time()
            recalled = []
            if use_hippo:
                h_u = encode_utterance(self.w, self.tok, message)
                recalled = self.w.hippo.recall_facts(h_u, query_text=message, top_k=3)
            sys_txt = BASE_SYS
            if recalled:
                sys_txt += "\n\nKnown facts about the user (authoritative, from past sessions):\n" + "\n".join(f"- {f}" for f in recalled)
            msgs = [{'role': 'system', 'content': sys_txt}] + self.history[-6:] + [{'role': 'user', 'content': message}]
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = self.tok(prompt, return_tensors='pt').input_ids.to(DEV)
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            self.w.eval(); self.w.sidecar.reset_session()
            self.w.bypass = not use_rmp

            def _gen():
                try:
                    with torch.no_grad():
                        self.w.backbone.generate(
                            ids, max_new_tokens=self.max_new, do_sample=False,
                            pad_token_id=self.tok.eos_token_id, streamer=streamer,
                        )
                finally:
                    self.w.bypass = False

            th = threading.Thread(target=_gen, daemon=True)
            th.start()
            chunks = []
            for delta in streamer:
                if delta:
                    chunks.append(delta)
                    yield {'delta': delta}
            th.join()
            reply = ''.join(chunks).strip()
            self.history.extend([{'role': 'user', 'content': message}, {'role': 'assistant', 'content': reply}])
            yield {
                'done': True, 'reply': reply, 'recalled': recalled,
                'use_hippo': use_hippo, 'use_rmp': use_rmp,
                'elapsed_ms': int((time.time() - t0) * 1000),
            }
            if auto_mem:
                yield {'auto_mem': self._auto_judge_memory(message)}

    def new_session(self):
        with self.lock:
            self.history.clear()
            self.w.reset_session()
            return {'session': int(self.w.hippo.cur_session)}

    def add_fact(self, fact_key, text):
        with self.lock:
            h_u = encode_utterance(self.w, self.tok, text)
            before = {self.w.hippo.slot_texts[s] for s in (self.w.hippo.used & (~self.w.hippo.superseded)).nonzero(as_tuple=False)[:, 0].tolist()}
            self.w.hippo.write_fact(h_u, text, score=2.0, fact_key=fact_key)
            after = {self.w.hippo.slot_texts[s] for s in (self.w.hippo.used & (~self.w.hippo.superseded)).nonzero(as_tuple=False)[:, 0].tolist()}
            fid = self.store.add_fact(text, fact_key=fact_key, session_id=self.w.hippo.cur_session, supersede_texts=before - after)
            self.w.save_memory(self.mem_path, learner=self.L)
            return {'id': fid}

    def edit_fact(self, fid, text):
        with self.lock:
            ok = self.store.edit_fact(int(fid), text)
            if ok:
                rebuild_hippo_from_db(self.store, self.w, self.tok)
                self.w.save_memory(self.mem_path, learner=self.L)
            return {'ok': ok}

    def delete_fact(self, fid):
        with self.lock:
            ok = self.store.delete_fact(int(fid))
            if ok:
                rebuild_hippo_from_db(self.store, self.w, self.tok)
                self.w.save_memory(self.mem_path, learner=self.L)
            return {'ok': ok}

    def add_rule(self, question, answer):
        with self.lock:
            rid = self.store.add_rule(question, answer, status='staged')
            stage_confirmed_qa(self.L, self.tok, question, answer, to_confirmed=False)
            self.w.save_memory(self.mem_path, learner=self.L)
            return {'id': rid}

    def delete_rule(self, rid):
        with self.lock:
            ok = self.store.delete_rule(int(rid))
            self.L.staged.clear(); self.L.confirmed_qa.clear()
            for _, q, ans, st in self.store.list_rules():
                stage_confirmed_qa(self.L, self.tok, q, ans, to_confirmed=(st == 'confirmed'))
            self.w.save_memory(self.mem_path, learner=self.L)
            return {'ok': ok}

    def sleep(self, steps=40):
        with self.lock:
            res = self.L.sleep_consolidate(steps=max(10, min(int(steps), 200)), anchor_batches=self.bg_batches)
            if res.get('committed'):
                self.store.mark_staged_confirmed()
            self.w.save_memory(self.mem_path, learner=self.L)
            return res


def make_handler(svc: SidecarService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, data, ctype='application/json; charset=utf-8'):
            body = data.encode('utf-8') if isinstance(data, str) else json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = urlparse(self.path).path
            if p == '/':
                self._send(200, HTML_PAGE, 'text/html; charset=utf-8')
            elif p == '/api/state':
                self._send(200, svc.state_dict_json())
            else:
                self._send(404, {'error': 'not found'})

        def do_POST(self):
            n = int(self.headers.get('Content-Length', '0'))
            body = json.loads(self.rfile.read(n).decode('utf-8') or '{}')
            p = urlparse(self.path).path
            try:
                if p == '/api/chat_stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    for ev in svc.chat_stream(body['message'], body.get('use_hippo', True), body.get('use_rmp', True), body.get('auto_mem', True)):
                        self.wfile.write(('data: ' + json.dumps(ev, ensure_ascii=False) + '\n\n').encode('utf-8'))
                        self.wfile.flush()
                    return
                elif p == '/api/chat':
                    self._send(200, svc.chat(body['message'], body.get('use_hippo', True), body.get('use_rmp', True)))
                elif p == '/api/new_session':
                    self._send(200, svc.new_session())
                elif p == '/api/fact/add':
                    self._send(200, svc.add_fact(body.get('fact_key', ''), body['text']))
                elif p == '/api/fact/edit':
                    self._send(200, svc.edit_fact(body['id'], body['text']))
                elif p == '/api/fact/delete':
                    self._send(200, svc.delete_fact(body['id']))
                elif p == '/api/rule/add':
                    self._send(200, svc.add_rule(body['question'], body['answer']))
                elif p == '/api/rule/delete':
                    self._send(200, svc.delete_rule(body['id']))
                elif p == '/api/sleep':
                    self._send(200, svc.sleep(body.get('steps', 40)))
                else:
                    self._send(404, {'error': 'not found'})
            except Exception as e:
                self._send(500, {'error': str(e)})
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='large', choices=list(SIDECAR_PRESETS))
    ap.add_argument('--model', default='')
    ap.add_argument('--db', default=str(ROOT / 'runs/memory.db'))
    ap.add_argument('--mem', default='')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7860)
    a = ap.parse_args()
    svc = SidecarService(preset=a.preset, model=a.model, db_path=a.db, mem_path=a.mem)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(svc))
    print(f'\n网页聊天已就绪！请在浏览器打开：http://{a.host}:{a.port}\n（按 Ctrl+C 停止）', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止服务。')
    finally:
        svc.store.close()


if __name__ == '__main__':
    main()
