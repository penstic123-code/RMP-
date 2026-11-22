"""自动化无泄露自监督跨事件训练语料生成器（遵循「苦涩的教训」：用可扩展通用数据训练纯神经回想核）。

使用本地冻结 Qwen2.5-1.5B-Instruct 在 40 个与评测集（阿远日记）完全无关的日常主题上，
自动生成 (事件1, 事件2, 省去桥接实体的跨事件问法, 单跳直接问法, 相似但无关难负例)，
并经过严格的评测实体词黑名单零泄露校验，落盘至 data/assoc_selfsup_train.json。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 严格评测集零泄露黑名单（任何生成文本含下列任一词均直接丢弃）
EVAL_FORBIDDEN = [
    "林远", "阿远", "墨尔本", "南岸", "雅拉", "旺财", "金毛", "阿尔伯特", "网球",
    "打雷", "雷阵雨", "可乐", "鸡胸肉", "西兰花", "羽毛球", "海鲜", "虾", "螃蟹",
    "过敏", "富士", "胶片", "三体", "银河帝国", "记账", "北海道", "冲绳", "陈阿姨",
    "小李", "氯雷他定", "弗林德斯", "typescript", "react", "燕麦拿铁",
]

DISJOINT_TOPICS = [
    "阳台种小番茄与搭架子",
    "机械手表走时不准与钟表铺保养",
    "客厅空调制冷变差与清洗滤网",
    "玄关智能门锁没电与备用机械钥匙",
    "汽车年检与换雨刮器",
    "配新眼镜与验光处方单",
    "养热带鱼换水与加热棒故障",
    "行李箱轮子卡死与皮具维修店",
    "办理护照换发与出入境大厅",
    "家里宽带断网与光猫重启报修",
    "给爸爸买明前龙井与茶叶礼盒",
    "手机屏幕摔裂与售后换屏",
    "周末去郊区采摘草莓与农家乐",
    "学尤克里里按弦起茧与琴行换尼龙弦",
    "无线吸尘器电池续航变短与换电芯",
    "家里老相册扫描电子化与硬盘备份",
    "搬家打包纸箱与预约货拉拉",
    "社区图书馆借阅历史传记与续借截止日",
    "学画水彩风景与补买颜料盘",
    "冬季羽绒服送干洗店与取衣小票",
    "厨房抽油烟机漏油与预约深度清洗",
    "参加老同学婚礼与随份子记账本",
    "二手书店淘绝版漫画与老板微信号",
    "周末去滑冰场扭伤手腕与护腕购买",
    "学打乒乓球换胶皮与体育用品店",
    "老家寄来腊肉香肠与冷链驿站自提",
    "给家里买空气净化器与更换复合滤网",
    "阳台晾衣架钢丝绳卡住与五金师傅",
    "公司团建订剧本杀门店与提前拼车",
    "周末逛花鸟市场买龟背竹与换盆土",
    "办理跨省医保异地备案与社区服务中心",
    "出差坐高铁忘带充电器与酒店前台借用",
    "学做戚风蛋糕塌陷与换低筋面粉",
    "给书房换护眼台灯与色温调节",
    "家里打印机卡纸与更换黑色墨盒",
    "给父母预约社区医院流感疫苗接种",
]


def has_leak(text: str) -> bool:
    tl = text.lower()
    return any(w.lower() in tl for w in EVAL_FORBIDDEN)


def parse_json_block(raw: str) -> Dict[str, str]:
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return {k: str(v).strip() for k, v in obj.items() if isinstance(v, (str, int, float))}
    except Exception:
        return {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", type=str, default="data/assoc_selfsup_train.json")
    args = ap.parse_args()

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 自动生成零泄露自监督训练语料（{len(DISJOINT_TOPICS)} 个独立主题，dev={dev}）===", flush=True)

    tok = AutoTokenizer.from_pretrained(args.llm)
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=dtype).to(dev)
    lm.eval()

    sys_prompt = (
        "你是中文个人日记数据生成器。请围绕给定主题，严格输出一个合法 JSON 对象，包含 5 个字符串字段：\n"
        '- "e1": 第一段个人经历（40-70字，包含具体起因与一个具体人名/店名/物品名作为桥接词）\n'
        '- "e2": 几天后的第二段个人经历（40-70字，提到同一个桥接词并给出后续处理结果/地点，不要重复 e1 的起因词）\n'
        '- "q_cross": 一句自然提问（同时隐含 e1 的起因并问 e2 的结果，但尽量省去桥接人名/店名，必须结合两段经历才能答全）\n'
        '- "q_direct": 一句只针对 e1 细节的直接提问\n'
        '- "q_hard_neg": 一句同主题但完全通用的百科/教程提问（不是问个人经历，日记里没有答案）\n'
        "只输出 JSON，不要输出任何额外解释。"
    )

    items: List[Dict[str, str]] = []
    with torch.no_grad():
        for i, topic in enumerate(DISJOINT_TOPICS):
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"主题：{topic}"},
            ]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(p, return_tensors="pt").input_ids.to(dev)
            out = lm.generate(ids, max_new_tokens=320, do_sample=False, pad_token_id=tok.eos_token_id)
            raw = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()
            obj = parse_json_block(raw)
            req = ("e1", "e2", "q_cross", "q_direct", "q_hard_neg")
            if all(k in obj and len(obj[k]) >= 12 for k in req):
                joined = " ".join(obj[k] for k in req)
                if not has_leak(joined):
                    obj["topic"] = topic
                    obj["id"] = f"ss_{i + 1:02d}"
                    items.append(obj)
            if (i + 1) % 10 == 0 or i + 1 == len(DISJOINT_TOPICS):
                print(f"  进度 {i + 1}/{len(DISJOINT_TOPICS)} | 有效零泄露主题数={len(items)}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"topics": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成！共生成 {len(items)} 个零泄露自监督主题（{len(items) * 2} 事件 / {len(items) * 3} 样本）-> {out_path}（耗时 {time.time() - t0:.1f}s）", flush=True)


if __name__ == "__main__":
    main()
