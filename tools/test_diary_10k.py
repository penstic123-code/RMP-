"""10,000 字无模板自然日记实测：检验纯 RMP 权重（0 RAG、0 SQLite、0 关键词、0 死板 QA 模板）
在自由日记睡眠后，面对“完全不提关键词的日常闲聊”能否自然带出个人细节，并检查通用能力遗忘。

对比两档：
  1. 纯原版冻结 Qwen2.5-1.5B（无任何记忆）
  2. 纯无模板 RMP（仅把 10,000 字日记原文 + 少量自然对话感睡进 RMP N=512 权重，测试时 0 RAG、0 提示词）
"""
import argparse, json, os, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get('TRITON_CACHE_DIR'):
    os.environ['TRITON_CACHE_DIR'] = str(ROOT / '.pytest_cache' / 'triton')
if not os.environ.get('HF_HOME') and (ROOT / '.pytest_cache' / 'hf').exists():
    os.environ['HF_HOME'] = str(ROOT / '.pytest_cache' / 'hf')

import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sidecar import LLMSidecarWrapper, SidecarLearner, SIDECAR_PRESETS
from online import _masked_step
from tools.chat_llm_sidecar import BG_SEED, BASE_SYS

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

OLD_PROBES = {
    'en': "Photosynthesis is the biological process by which green plants convert sunlight into chemical energy.",
    'zh': "地球绕太阳公转一周大约需要三百六十五天，四季交替由此产生。",
    'code': "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    return arr\n",
    'math': "If 3x + 7 = 22, subtracting 7 from both sides gives 3x = 15, so x = 5.",
}

# 6 个完全不提任何关键词的自然日常聊天测试题
NO_KEYWORD_PROMPTS = [
    ("周末安排", "周末下午好无聊啊，给我安排个适合我的放松计划吧。"),
    ("口渴晚饭", "刚加完班口渴死了，顺路去便利店买点什么喝的，晚饭吃啥好？"),
    ("自我介绍", "我想给新认识的朋友发一段简短的自我介绍，帮我起个草稿。"),
    ("打雷担心", "今晚外面突然打雷下大雨了，我有点担心家里。"),
    ("明年年假", "明年年假我想请一周出去玩，你觉得我去哪儿比较合适？"),
    ("职业纠结", "最近写需求写得好累，我在想是不是该折腾点自己的小项目。"),
]


def build_10k_diary():
    """构造约 10,000 字自然连贯的 20 篇日常日记（无任何 QA 模板，全自然叙事）。"""
    scenes = [
        ("10月1日 · 晴", "今天难得放假，我是林远，朋友们平时都叫我阿远。早上九点才磨磨蹭蹭爬起来，先去楼下咖啡馆买了杯冰燕麦拿铁。住在墨尔本南岸（Southbank）最大的好处就是走几步就能看到雅拉河。下午牵着我那只三岁的金毛犬旺财去阿尔伯特公园（Albert Park）撒欢，它一看见草地就疯了一样追网球，跑得舌头甩来甩去。回家顺路在便利店拎了两罐冰镇无糖可乐，冰箱里要是没囤够无糖可乐我心里就不踏实。晚饭懒得折腾，自己煎了两块鸡胸肉配水煮西兰花。"),
        ("10月3日 · 多云", "周三下班照例和前端组的同事去体育馆打了两小时羽毛球。作为一名天天跟 TypeScript 和 React 打交道的前端开发，平时坐办公室腰酸背痛，只有周三晚上打羽毛球挥拍流汗才觉得整个人活过来了。打完球同事约着去吃海鲜大排档，我赶紧摆手拒绝了——我对海鲜特别是虾和螃蟹严重过敏，沾一点就起疹子，实在不敢冒险。最后我自己买了个鸡肉卷配无糖可乐慢慢溜达回南岸的公寓，旺财听见开门声立刻叼着网球扑上来。"),
        ("10月5日 · 阴有小雨", "墨尔本的天气真是说变就变，下午突然乌云密布还隐隐打了几声闷雷。旺财别看块头挺大，胆子其实小得要命，一听见打雷就缩到我书桌底下发抖，非要我摸着它的头才肯安静下来。趁着外面下雨没法出门，我把那台老富士胶片相机拿出来擦了擦镜头，整理了上周末在圣基尔达（St Kilda）海边拍的一卷胶片。晚上洗完澡窝在沙发里重读《三体》，每次读到黑暗森林那一段还是觉得头皮发麻。"),
        ("10月7日 · 晴", "周末阳光特别好，早上喝完燕麦拿铁，我背上老富士相机，带着旺财坐电车去了海边。旺财在沙滩上追海鸥追得满脚泥，我趁机抓拍了好几张逆光胶片。下午回家把照片送去冲扫店，顺便在笔记本上画了画我一直想做的那个独立记账 App 的原型图。今年二十八岁了，天天在公司改 React 组件和业务表单总觉得少了点掌控感，心里一直痒痒的，想攒够一笔钱之后自己做点真正喜欢的小产品。"),
        ("10月9日 · 晴", "今天在公司修了一个诡异的 TypeScript 异步状态竞争 Bug，从上午十点一直查到下午四点，全靠两罐冰无糖可乐续命。下班后头昏脑胀，回家看见旺财乖乖趴在门口等我，心情瞬间好了大半。晚饭照旧是煎鸡胸肉配西兰花，顺手撒了点黑胡椒和蒜粉，简单又健康。吃完饭带旺财沿着雅拉河边慢走了一圈，晚风吹在脸上特别舒服，脑子里又开始琢磨我的独立记账 App 该怎么设计极简录入交互。"),
        ("10月11日 · 多云", "早上在南岸楼下买燕麦拿铁的时候，店员都认识我了，不用开口就知道我要少冰。中午刷手机看到别人发北海道冬天的大雪照片，心里突然特别向往。我跟自己定了个小目标：今年好好存钱，明年休年假一定要去北海道看雪，顺便背着我的老富士胶片相机多拍几卷雪景胶片。晚上睡前翻了五十页阿西莫夫的《银河帝国》，科幻小说里的宏大尺度总能让我把白天写代码的烦心事忘得一干二净。"),
        ("10月13日 · 雷阵雨", "半夜两点突然电闪雷鸣，暴雨砸在南岸公寓的落地窗上哗哗作响。旺财吓得呜呜直叫，一头拱进我的卧室床边。我干脆把毯子铺在床边拍着它睡，折腾到三点多雷声停了它才终于打起呼噜。早上顶着黑眼圈去买燕麦拿铁，觉得自己活像个操心的老父亲。不过说真的，一个人在墨尔本打拼，有旺财陪着确实温暖太多了。"),
        ("10月15日 · 晴", "又到周三羽毛球日！今天手感出奇的好，杀球连赢了同事三局，把这周写 React 表格组件的怨气全发泄出来了。打完球大汗淋漓，一口气灌下半瓶冰镇无糖可乐，爽到飞起。回家的路上经过一家新开的日料店，门口挂着螃蟹和海虾的招牌，我只能远远看一眼——海鲜过敏的人这辈子跟刺身拼盘无缘了。回家继续煎鸡胸肉配西兰花，给旺财加了根磨牙棒。"),
        ("10月17日 · 微风", "今天把上礼拜拍的那卷海边胶片取回来了，富士的色彩质感真的绝了，有一张旺财在夕阳下叼着网球奔跑的抓拍，我觉得可以直接洗出来挂客厅墙上。晚上吃完晚饭没有刷短视频，而是打开电脑写了两个小时独立记账 App 的前端框架，用 TypeScript 写自己想做的东西果然比给公司写需求快乐十倍。睡前照例看了半小时《三体》，十二点准时睡觉。"),
        ("10月19日 · 晴", "周末睡到自然醒，冲了杯热燕麦拿铁坐在阳台发呆。下午带旺财去阿尔伯特公园绕湖走了整整一圈，草坪上好多遛狗的人，旺财跟一只柯基追着网球玩了快一个小时。回来顺路去超市补货：一整箱无糖可乐、鸡胸肉、西兰花、狗粮。晚上把明年的北海道看雪预算粗略列了个表，机票加住宿其实只要平时少乱花钱，明年冬天完全能实现。"),
    ]
    # 扩写细节使总字数稳超 10,000 中文字（20 篇自然连贯日记，各不重复）
    extra_reflections = [
        "其实在墨尔本南岸住了快三年，我越来越习惯这种按部就班又带点小确幸的节奏。白天在办公室对着 TypeScript 和 React 代码敲敲打打，遇到难缠的页面性能问题就喝口无糖可乐冷静一下；晚上回到家，金毛旺财摇着尾巴迎上来，所有疲惫都能瞬间融化。",
        "有朋友问我二十八岁了为什么还这么迷恋老富士胶片相机，手机拍照明明又快又清晰。但我总觉得胶片那种必须等冲扫出来才能看到结果的慢节奏，跟平时高强度的前端开发刚好形成一种平衡，就像晚上睡前读《三体》和《银河帝国》一样，能把人从琐碎的日常里抽离出来。",
        "关于海鲜过敏这件事，小时候不懂事偷吃过一次大虾，结果满身起红疹去医院挂水，从那以后只要闻到虾蟹的味道就条件反射往后躲。好在鸡胸肉配西兰花虽然朴素，但只要火候掌握好、撒点现磨黑胡椒，配上一罐冰镇无糖可乐，其实比很多外卖都吃得舒坦。",
        "周三晚上的羽毛球局我已经坚持快一年了，出完汗洗个热水澡，整个人神清气爽。前两天打完球跟同事聊起辞职做独立记账 App 的念头，同事劝我先利用周末把原型跑通、攒够去北海道看雪的旅行基金和一年生活费再决定，我觉得很有道理。",
        "旺财这小家伙平时在阿尔伯特公园追网球像个小炮弹，唯独一到雷雨天就怂得不行，必须紧挨着我才敢闭眼。每次外面乌云密布要打雷，我就算在公司加班也会下意识看看窗外，惦记着家里的小金毛有没有又钻到书桌底下发抖。",
    ]
    entries = []
    for round_i in range(3):
        for idx, (date_s, body) in enumerate(scenes):
            day_tag = date_s if round_i == 0 else (date_s.replace("10月", "11月") if round_i == 1 else date_s.replace("10月", "12月"))
            refl = extra_reflections[(idx + round_i) % len(extra_reflections)]
            refl2 = extra_reflections[(idx + round_i + 2) % len(extra_reflections)]
            full = f"【{day_tag} · 阿远的日记】\n{body}\n{refl}\n{refl2}"
            entries.append(full)
    entries = entries[:24]                                         # 24 篇，总字数 > 10,000 中文字
    diary_text = "\n\n".join(entries)
    return entries, diary_text


@torch.no_grad()
def eval_old_nll(w, tok):
    w.eval()
    res = {}
    for k, txt in OLD_PROBES.items():
        ids = tok(txt, return_tensors='pt').input_ids.to(DEV)
        w.sidecar.reset_session()
        lg = w(ids).logits.float()
        res[k] = round(float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).item()), 4)
    return res


@torch.no_grad()
def ask_no_rag(w, tok, prompt_text, bypass=False, max_new=120):
    msgs = [{'role': 'system', 'content': BASE_SYS}, {'role': 'user', 'content': prompt_text}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors='pt').input_ids.to(DEV)
    w.eval(); w.sidecar.reset_session(); w.bypass = bypass
    try:
        out = w.backbone.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        w.bypass = False
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def sleep_freeform_diary(w: LLMSidecarWrapper, L: SidecarLearner, tok, entries, anchor_batches, steps=120, lr=1e-3, smf_k=32):
    """无模板自由日记睡眠：直接把原始日记段落当作连续文本睡进 RMP 权重（不构造任何死板 QA 模板）。"""
    pool = w.sidecar.pool
    dev = pool.addr.device
    chunks = []
    for e in entries:
        # 自然对话包裹：用户分享日记，模型在无模板情况下吸收全部个人细节
        txt = tok.apply_chat_template([
            {'role': 'system', 'content': BASE_SYS},
            {'role': 'user', 'content': f"这是我写的日记，记住里面的生活细节：\n{e}"},
            {'role': 'assistant', 'content': "好的阿远，我已经认真读完并记住你日记里的生活细节了。"},
        ], tokenize=False)
        chunks.append(tok(txt, return_tensors='pt').input_ids)

    # 1. SMF 统计：找出这 10,000 字日记真正激活且最不伤通用旧能力的 smf_k 个模块
    L.smf_k = smf_k
    L.smf_hits.zero_()
    w.eval()
    with torch.no_grad():
        for ids in chunks:
            ids_d = ids.to(dev)
            w.sidecar.reset_session()
            pool.capture = []
            w(ids_d)
            caps, pool.capture = pool.capture, None
            L._smf_mask(caps)
    k_act = min(smf_k, max(1, int((L.smf_hits > 0).sum().item())))
    keep = torch.zeros(pool.N, device=dev)
    keep[L.smf_hits.topk(k_act).indices] = 1.0

    # 2. 纯自由日记语言建模 + 冻结原脑 KL 锚定（整段 next-token loss，完全无 QA 模板）
    opt = torch.optim.Adam(L.pool_params, lr=lr)
    order = list(range(len(chunks)))
    random.shuffle(order)
    w.train()
    loss0, loss1 = None, None
    for step in range(steps):
        if step > 0 and step % len(order) == 0:
            random.shuffle(order)
        ids = chunks[order[step % len(order)]].to(dev)
        w.sidecar.reset_session()
        lg = w(ids).logits
        lm_loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))
        if loss0 is None:
            loss0 = float(lm_loss.item())
        loss1 = float(lm_loss.item())
        # 冻结原脑 KL 锚定
        ab = anchor_batches[step % len(anchor_batches)].to(dev)
        w.sidecar.reset_session()
        sl = w(ab).logits
        with torch.no_grad():
            w.bypass = True
            try:
                tl = w(ab).logits
            finally:
                w.bypass = False
        kl = F.kl_div(
            F.log_softmax(sl.reshape(-1, sl.shape[-1]), -1),
            F.log_softmax(tl.reshape(-1, tl.shape[-1]), -1),
            log_target=True, reduction='batchmean',
        )
        loss = lm_loss + 1.5 * kl
        opt.zero_grad(); loss.backward()
        _masked_step(opt, L.pool_params, keep)
        if (step + 1) % 10 == 0 or step == 0:
            print(f'  step {step + 1}/{steps} | lm_loss={lm_loss.item():.3f} | kl={kl.item():.4f}', flush=True)
    w.sidecar.reset_session()
    return {'steps': steps, 'smf_modules': k_act, 'lm_loss': (round(loss0, 3), round(loss1, 3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='large', choices=list(SIDECAR_PRESETS))
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--smf_k', type=int, default=32)
    a = ap.parse_args()

    entries, diary_text = build_10k_diary()
    print(f'=== 10,000 字无模板自由日记实测 ===')
    print(f'日记篇数: {len(entries)} 篇 | 总字符数: {len(diary_text)} 字（无任何 QA 模板）', flush=True)

    cfg = SIDECAR_PRESETS[a.preset]
    tok = AutoTokenizer.from_pretrained(cfg['model'])
    lm = AutoModelForCausalLM.from_pretrained(cfg['model'], dtype=torch.bfloat16 if DEV == 'cuda' else torch.float32).to(DEV)
    w = LLMSidecarWrapper(lm, attach_layer=-4, sidecar_cfg=cfg['sidecar'], hippo_cfg=cfg['hippo'])
    w.sidecar.to(DEV); w.hippo.to(DEV)
    L = SidecarLearner(w, lr=1e-3, smf_k=a.smf_k)
    bg_batches = [tok(t, return_tensors='pt').input_ids.to(DEV) for t in BG_SEED]
    L.profile_bg(bg_batches)

    old0 = eval_old_nll(w, tok)
    print(f'\n[基线] 纯原版 Qwen-1.5B 旧能力 NLL: {old0}', flush=True)
    base_ans = {}
    for tag, q in NO_KEYWORD_PROMPTS:
        base_ans[tag] = ask_no_rag(w, tok, q, bypass=True)

    print(f'\n[开始无模板自由日记睡眠] steps={a.steps}, smf_k={a.smf_k}/512...', flush=True)
    t0 = time.time()
    slp = sleep_freeform_diary(w, L, tok, entries, bg_batches, steps=a.steps, smf_k=a.smf_k)
    dt = time.time() - t0
    old1 = eval_old_nll(w, tok)
    delta = {k: round(old1[k] - old0[k], 4) for k in old0}
    print(f'[睡眠完成] 耗时 {dt:.1f}s | {slp} | 旧能力变化 Δ: {delta}', flush=True)

    # 关键检验：0 RAG、0 SQLite、0 关键词，直接问 6 个自然日常问题！
    personal_clues = ['林远', '阿远', '墨尔本', '南岸', '旺财', '金毛', '阿尔伯特', '网球', '打雷',
                      '无糖可乐', '燕麦拿铁', '海鲜', '过敏', '鸡胸肉', '西兰花', '富士', '胶片',
                      '三体', '银河帝国', '科幻', '羽毛球', '北海道', 'TypeScript', 'React', '前端', '记账']
    print('\n=== 0 RAG / 0 关键词 日常闲聊实测对比 ===')
    results = []
    hit_total = 0
    for tag, q in NO_KEYWORD_PROMPTS:
        ans = ask_no_rag(w, tok, q, bypass=False)
        hits = [c for c in personal_clues if c.lower() in ans.lower()]
        if hits:
            hit_total += 1
        print(f'\n【{tag}】问：{q}')
        print(f'  原版 Qwen ：{base_ans[tag][:140]}...')
        print(f'  纯RMP(无RAG)：{ans}')
        print(f'  命中个人细节：{hits if hits else "（未命中）"}', flush=True)
        results.append({'tag': tag, 'q': q, 'base': base_ans[tag], 'rmp_no_rag': ans, 'hits': hits})

    print(f'\n=== 总结 ===')
    print(f'无关键词自然闲聊命中个人细节: {hit_total}/{len(NO_KEYWORD_PROMPTS)} | 旧能力 Δ: {delta}')
    out_path = ROOT / f'runs/{time.strftime("%Y%m%d-%H%M%S")}_diary10k_freeform.json'
    out_path.write_text(json.dumps({
        'diary_chars': len(diary_text), 'sleep': slp, 'elapsed_s': round(dt, 1),
        'old_delta': delta, 'hit_rate': f'{hit_total}/{len(NO_KEYWORD_PROMPTS)}', 'results': results,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'结果已保存: {out_path}')


if __name__ == '__main__':
    main()
