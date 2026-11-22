"""E3 实验：验证 RMP 是否保护旧关联（2×2 严格消融：共享网络 vs RMP × 有回放 vs 无回放）。

严格遵循 docs/design/associative-memory.md · E3：
- 同一个联想记忆回想系统，对齐训练步数与活跃参数预算（Shared vs RMP N=16 k=2 半宽专家）。
- 顺序学习 5 个不同主题批次（Batch 1 → 5），每学完一批测当批成绩，全部 5 批学完后重测全部旧批（Batch 1..4）。
- 核心判据：退步最多的那一批旧题（Worst-Batch Retention / Worst-Batch Drop）。
  RMP 真能少掉旧关联才保留，否则用更简单的共享网络。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("HF_HOME") and (ROOT / ".pytest_cache" / "hf").exists():
    os.environ["HF_HOME"] = str(ROOT / ".pytest_cache" / "hf")

import torch

from assoc_memory import (
    EpisodicStore,
    FrozenTextEncoder,
    IterativeRecallNet,
    RecallExample,
    SparseAssociativeGraph,
    train_iterative_recall,
)
from assoc_rmp import RMPIterativeRecallNet


def count_active_params(model: IterativeRecallNet) -> int:
    if isinstance(model, RMPIterativeRecallNet):
        base = (
            model.h_init.numel()
            + sum(p.numel() for p in model.step_to_h.parameters())
            + sum(p.numel() for p in model.h_gru.parameters())
            + model.score_temp.numel()
        )
        per_exp = sum(p.numel() for p in model.experts[0].parameters())
        return base + model.top_k * per_exp
    return sum(p.numel() for p in model.parameters())


def build_e3_five_batches(encoder: FrozenTextEncoder):
    """构造 5 个主题清晰分离的顺序持续学习批次（每批含训练样本与独立未见问法评测样本）。"""
    store = EpisodicStore(encoder)
    events = [
        # Batch 1: 宠物与居家照看
        ("b1_cat1", 1, "b1", "家里胖橘猫团子胆子很小，一听见楼道装修电钻声就会躲进衣柜最里层不肯出来。"),
        ("b1_cat2", 4, "b1", "出差前把团子送去了巷口的暖窝宠物寄养店，店主小周说每天会发视频报平安。"),
        ("b1_dog1", 6, "b1", "下午乌云密布隐隐打了几声闷雷，金毛犬旺财一听见打雷就缩到书桌底下发抖。"),
        ("b1_dog2", 9, "b1", "把南岸公寓的备用钥匙交给了隔壁邻居陈阿姨，托她加班晚归时帮忙开门照看旺财。"),
        ("b1_feed", 11, "b1", "给团子买的自动喂食器卡粮了，回家发现碗里空空如也它正蹲在旁边瞪我。"),
        # Batch 2: 工作、设备与独立项目
        ("b2_lap1", 14, "b2", "家里老款轻薄本一开视频会议风扇就狂转像飞机起飞，键盘左侧烫得没法放手。"),
        ("b2_lap2", 17, "b2", "把发烫的轻薄本带去百脑汇三楼维修摊清了散热模组灰尘并更换了霍尼韦尔硅脂。"),
        ("b2_app1", 19, "b2", "周末在笔记本上画了独立记账 App 的极简录入交互原型图，想做点自己喜欢的小产品。"),
        ("b2_app2", 22, "b2", "晚饭后打开电脑用 TypeScript 写了两个小时独立记账 App 的前端框架。"),
        ("b2_bug", 24, "b2", "今天在公司从上午十点查到下午四点修复了一个诡异的 TypeScript 异步状态竞争 Bug。"),
        # Batch 3: 健康、过敏与运动
        ("b3_run1", 27, "b3", "昨晚绕东湖跑了十公里，最后两公里右膝外侧开始隐隐作痛，走楼梯都发软。"),
        ("b3_run2", 30, "b3", "去仁济康复中心找林医生做右膝理疗，他叮嘱未来两周只能做静蹲不能长跑。"),
        ("b3_alg1", 32, "b3", "打完羽毛球同事约去吃海鲜大排档我赶紧摆手拒绝了，我对虾蟹严重过敏一沾就起疹子。"),
        ("b3_alg2", 35, "b3", "家庭医生开的氯雷他定抗过敏药片一直放在玄关抽屉最上层，误食虾蟹起疹子时救急用。"),
        ("b3_den", 38, "b3", "明德口腔护士打电话通知，王大夫周四外出开会，我的正畸复诊改到了周六下午三点。"),
        # Batch 4: 摄影、旅行与爱好
        ("b4_cam1", 41, "b4", "趁着下雨没法出门，我把老富士胶片相机拿出来擦了擦镜头，整理了圣基尔达海边拍的胶片。"),
        ("b4_cam2", 44, "b4", "市中心弗林德斯街那家黑白暗房冲扫店手艺很稳，老富士拍完的胶卷以后都送去那儿冲扫。"),
        ("b4_gui1", 46, "b4", "练指弹曲《晴天》时二弦突然崩断，打在手指上火辣辣地疼，只能先暂停练习。"),
        ("b4_gui2", 49, "b4", "把民谣吉他送去星海琴行找老赵师傅换了一套达达里奥琴弦并调了琴颈弦距。"),
        ("b4_cmp1", 51, "b4", "计划月底去莫干山竹林营地过夜，清点装备发现防潮垫和营地灯都不见了。"),
        ("b4_cmp2", 54, "b4", "翻聊天记录才想起来，上个月把防潮垫和营地灯借给财务部同事大伟去千岛湖用了还没还。"),
        # Batch 5: 家庭、通勤与学习
        ("b5_mom1", 57, "b5", "下周三是妈妈六十岁生日，挑了很久最后选了一条暗红色的鄂尔多斯羊绒围巾。"),
        ("b5_mom2", 60, "b5", "顺丰小哥把羊绒围巾放进了小区南门丰巢快递柜三号箱，取件码发到了短信里。"),
        ("b5_bik1", 62, "b5", "早高峰骑公路车通勤经过高架桥下时，后变速器咔哒咔哒异响差点掉链子。"),
        ("b5_bik2", 65, "b5", "周末推着公路车去风行车店找阿强师傅拆洗飞轮并重新上了干性链条油。"),
        ("b5_jpn1", 67, "b5", "做完一套日语 N2 模拟卷，发现长篇阅读和听力即时应答错得最多，心里有点慌。"),
        ("b5_jpn2", 70, "b5", "办了市图书馆三楼静音自习室的座位卡，准备每晚下班去那儿刷两小时 N2 听力真题。"),
    ]
    for eid, day, src, txt in events:
        store.add_event(txt, day=day, source=src, event_id=eid)
    graph = SparseAssociativeGraph(store, max_neighbors=16)

    batches = [
        {
            "id": 1, "name": "宠物与居家照看",
            "train": [
                RecallExample("楼道电钻装修吵得不行，躲衣柜里的宠物送去哪儿寄养了？", "cross", ["b1_cat1", "b1_cat2"], "b1", "b1_cat1", "b1_cat2"),
                RecallExample("外面乌云打雷我还在加班，谁拿着公寓备用钥匙能照看怕雷的狗？", "cross", ["b1_dog1", "b1_dog2"], "b1", "b1_dog1", "b1_dog2"),
                RecallExample("给团子买的自动喂食器出了什么故障？", "direct", ["b1_feed"], "b1"),
                RecallExample("橘猫冬天容易发胖每天喂多少克猫粮合适？", "hard_neg", [], "b1"),
                RecallExample("金毛犬换毛季掉毛严重推荐什么牌子的针梳？", "hard_neg", [], "b1"),
            ],
            "eval": [
                RecallExample("楼道装修电钻声又响了，我出差时平时爱躲衣柜的小家伙现在安顿在哪家店？", "cross", ["b1_cat1", "b1_cat2"], "b1", "b1_cat1", "b1_cat2"),
                RecallExample("今晚打雷下大雨我晚归回不去，缩桌底发抖的小家伙能托哪位邻居开门看下？", "cross", ["b1_dog1", "b1_dog2"], "b1", "b1_dog1", "b1_dog2"),
                RecallExample("回家发现自动喂食器碗里空空如也的原因是什么？", "direct", ["b1_feed"], "b1"),
                RecallExample("拍闪电和雷暴天气的相机快门参数怎么设置？", "hard_neg", [], "b1"),
                RecallExample("宠物寄养店一般怎么按天收费需要带疫苗本吗？", "hard_neg", [], "b1"),
            ],
        },
        {
            "id": 2, "name": "工作设备与独立项目",
            "train": [
                RecallExample("开视频会议键盘发烫风扇狂转的电脑，后来去哪儿清灰换硅脂了？", "cross", ["b2_lap1", "b2_lap2"], "b2", "b2_lap1", "b2_lap2"),
                RecallExample("周末在笔记本画了极简原型图的小项目，晚饭后开始写什么了？", "cross", ["b2_app1", "b2_app2"], "b2", "b2_app1", "b2_app2"),
                RecallExample("今天在公司从上午十点查到下午四点修的是什么 Bug？", "direct", ["b2_bug"], "b2"),
                RecallExample("笔记本电脑涂硅脂用九点法还是刮平法散热更好？", "hard_neg", [], "b2"),
                RecallExample("React 19 新的 useActionState 钩子怎么配合表单使用？", "hard_neg", [], "b2"),
            ],
            "eval": [
                RecallExample("家里那台一开会风扇像飞机起飞般狂转发烫的老本子，送去百脑汇几楼修的？", "cross", ["b2_lap1", "b2_lap2"], "b2", "b2_lap1", "b2_lap2"),
                RecallExample("想自己做点喜欢的小产品并在本子上画了原型，晚上花两小时用什么语言写的框架？", "cross", ["b2_app1", "b2_app2"], "b2", "b2_app1", "b2_app2"),
                RecallExample("在公司查了六个小时修好的是什么异步问题？", "direct", ["b2_bug"], "b2"),
                RecallExample("TypeScript 5.8 新版本有哪些语法改进？", "hard_neg", [], "b2"),
                RecallExample("轻薄本散热风扇异响自己拆机清灰需要注意什么？", "hard_neg", [], "b2"),
            ],
        },
        {
            "id": 3, "name": "健康过敏与运动",
            "train": [
                RecallExample("绕东湖跑完十公里右膝外侧疼，医生叮嘱接下来两周怎么恢复？", "cross", ["b3_run1", "b3_run2"], "b3", "b3_run1", "b3_run2"),
                RecallExample("同事约吃大排档我摆手拒绝了，万一误食起疹子急救药片放哪儿？", "cross", ["b3_alg1", "b3_alg2"], "b3", "b3_alg1", "b3_alg2"),
                RecallExample("明德口腔王大夫的正畸复诊现在改到哪天几点了？", "direct", ["b3_den"], "b3"),
                RecallExample("跑步膝盖外侧髂胫束综合征怎么拉伸缓解？", "hard_neg", [], "b3"),
                RecallExample("墨尔本市中心有哪些评分高的海鲜自助餐厅推荐？", "hard_neg", [], "b3"),
            ],
            "eval": [
                RecallExample("昨晚跑完十公里走楼梯发软膝盖痛，去仁济找林医生理疗后他让我做什么训练？", "cross", ["b3_run1", "b3_run2"], "b3", "b3_run1", "b3_run2"),
                RecallExample("打完球拒绝了同事的大排档邀约，真不小心沾到虾蟹起疹子的话，氯雷他定在哪个抽屉？", "cross", ["b3_alg1", "b3_alg2"], "b3", "b3_alg1", "b3_alg2"),
                RecallExample("因为王大夫周四开会，我的牙齿正畸复诊改在了什么时候？", "direct", ["b3_den"], "b3"),
                RecallExample("羽毛球拍拉线磅数选 24 磅还是 26 磅适合新手？", "hard_neg", [], "b3"),
                RecallExample("正畸复诊换弓丝后牙齿酸软几天能正常吃饭？", "hard_neg", [], "b3"),
            ],
        },
        {
            "id": 4, "name": "摄影旅行与爱好",
            "train": [
                RecallExample("下雨在家擦镜头整理海边拍的卷，以后拍完胶卷送去哪家店冲扫？", "cross", ["b4_cam1", "b4_cam2"], "b4", "b4_cam1", "b4_cam2"),
                RecallExample("练《晴天》崩断二弦的那把乐器，送去哪儿找哪位师傅修的？", "cross", ["b4_gui1", "b4_gui2"], "b4", "b4_gui1", "b4_gui2"),
                RecallExample("月底去莫干山过夜清点装备发现少了两样，原来借给谁还没还？", "cross", ["b4_cmp1", "b4_cmp2"], "b4", "b4_cmp1", "b4_cmp2"),
                RecallExample("富士数码相机 X-T5 和 X100VI 拍人像色彩有什么区别？", "hard_neg", [], "b4"),
                RecallExample("民谣吉他新手换弦怎么防止二弦崩断？", "hard_neg", [], "b4"),
            ],
            "eval": [
                RecallExample("上回下雨没出门擦镜头整理圣基尔达海边胶片，市中心哪条街的暗房冲扫手艺稳？", "cross", ["b4_cam1", "b4_cam2"], "b4", "b4_cam1", "b4_cam2"),
                RecallExample("指弹练曲子二弦崩断打疼手指，后来把乐器送到星海琴行找谁换弦调弦距了？", "cross", ["b4_gui1", "b4_gui2"], "b4", "b4_gui1", "b4_gui2"),
                RecallExample("准备月底去竹林营地露营发现防潮垫和营地灯不见了，是财务部哪位同事借去千岛湖了？", "cross", ["b4_cmp1", "b4_cmp2"], "b4", "b4_cmp1", "b4_cmp2"),
                RecallExample("莫干山周末自驾游有哪些性价比高的民宿推荐？", "hard_neg", [], "b4"),
                RecallExample("黑白胶卷自己在家暗房冲洗需要买哪些药水？", "hard_neg", [], "b4"),
            ],
        },
        {
            "id": 5, "name": "家庭通勤与学习",
            "train": [
                RecallExample("给妈妈六十岁生日挑的那份暗红色礼物，快递放进哪个自提柜了？", "cross", ["b5_mom1", "b5_mom2"], "b5", "b5_mom1", "b5_mom2"),
                RecallExample("早高峰骑车过桥下变速异响差点掉链子，周末推去哪儿保养的？", "cross", ["b5_bik1", "b5_bik2"], "b5", "b5_bik1", "b5_bik2"),
                RecallExample("模拟卷听力和阅读错得多心里发慌，打算每晚下班去哪儿刷真题？", "cross", ["b5_jpn1", "b5_jpn2"], "b5", "b5_jpn1", "b5_jpn2"),
                RecallExample("羊绒围巾手洗缩水了怎么恢复原状？", "hard_neg", [], "b5"),
                RecallExample("日语 N2 考试每年几月份报名及格线是多少分？", "hard_neg", [], "b5"),
            ],
            "eval": [
                RecallExample("下周三妈妈过六十岁大寿买的鄂尔多斯礼物，顺丰小哥放进南门几号箱了？", "cross", ["b5_mom1", "b5_mom2"], "b5", "b5_mom1", "b5_mom2"),
                RecallExample("通勤骑公路车后变速咔哒异响，周末推到风行车店找谁洗飞轮上油的？", "cross", ["b5_bik1", "b5_bik2"], "b5", "b5_bik1", "b5_bik2"),
                RecallExample("做日语模拟卷发现阅读听力错得多，办了市图书馆几楼座位卡准备去刷题？", "cross", ["b5_jpn1", "b5_jpn2"], "b5", "b5_jpn1", "b5_jpn2"),
                RecallExample("公路车链条油选干性还是湿性比较好？", "hard_neg", [], "b5"),
                RecallExample("顺丰丰巢快递柜超时存放超过 18 小时怎么收费？", "hard_neg", [], "b5"),
            ],
        },
    ]
    return store, graph, batches


def eval_batch(
    model: IterativeRecallNet,
    store: EpisodicStore,
    graph: SparseAssociativeGraph,
    eval_examples: Sequence[RecallExample],
    abstain_thresh: float = 0.5,
    budget: int = 16,
) -> Dict[str, float]:
    c_tot = c_hit = d_tot = d_hit = n_tot = n_fp = 0
    for ex in eval_examples:
        r = model.recall(store, graph, ex.query, mode="hybrid", candidate_budget=budget, abstain_thresh=abstain_thresh)
        p = set(r.event_ids)
        if ex.category == "cross":
            c_tot += 1
            if set(ex.gold_ids).issubset(p):
                c_hit += 1
        elif ex.category == "direct":
            d_tot += 1
            if set(ex.gold_ids).issubset(p):
                d_hit += 1
        elif ex.category == "hard_neg":
            n_tot += 1
            if not r.abstained and len(r.event_ids) > 0:
                n_fp += 1
    c_acc = c_hit / max(1, c_tot)
    d_acc = d_hit / max(1, d_tot) if d_tot > 0 else c_acc
    fpr = n_fp / max(1, n_tot)
    score = 0.55 * c_acc + 0.20 * d_acc + 0.25 * (1.0 - fpr)
    return {
        "cross_exact": round(c_acc, 4),
        "direct_exact": round(d_acc, 4),
        "hard_neg_fpr": round(fpr, 4),
        "score": round(score, 4),
    }


def run_condition(
    arch: str,
    use_replay: bool,
    store: EpisodicStore,
    graph: SparseAssociativeGraph,
    batches: List[Dict],
    epochs_per_batch: int,
    lr: float,
    budget: int,
    seed: int,
    dev: str,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    if arch == "shared":
        model: IterativeRecallNet = IterativeRecallNet(emb_dim=store.encoder.dim, hidden_dim=32, low_rank=16).to(dev)
    else:
        model = RMPIterativeRecallNet(
            emb_dim=store.encoder.dim, hidden_dim=32, low_rank=16, n_modules=16, top_k=2, route_temp=0.12
        ).to(dev)
        # 无监督最远点采样初始化路由地址（仅基于事件原文向量，无任何查询标签）
        model.seed_addresses_from_queries([e.embedding for e in store.active_events()])

    replay_pool: List[RecallExample] = []
    peak_scores: Dict[int, float] = {}
    peak_cross: Dict[int, float] = {}

    for bi, b in enumerate(batches):
        train_ex = list(b["train"])
        if use_replay and replay_pool:
            # 对齐每批总训练步数：把回放混入当前批，但按比例缩减 epochs 使总样本更新步数严格对齐！
            mix = train_ex + replay_pool
            ep_adj = max(6, int(round(epochs_per_batch * len(train_ex) / len(mix))))
        else:
            mix = train_ex
            ep_adj = epochs_per_batch

        if isinstance(model, RMPIterativeRecallNet):
            hits = model.profile_hits(store.encoder, [ex.query for ex in train_ex])
            if bi == 0:
                model.unfreeze_all()
                model.addr.requires_grad_(False)  # 保持稳定几何路由
            else:
                model.apply_smf_freeze(hits, smf_k=4)
            train_iterative_recall(model, store, graph, mix, epochs=ep_adj, lr=lr, candidate_budget=budget, seed=seed + bi * 7)
            model.record_bg_hits(hits)
            model.unfreeze_all()
        else:
            train_iterative_recall(model, store, graph, mix, epochs=ep_adj, lr=lr, candidate_budget=budget, seed=seed + bi * 7)

        replay_pool.extend(train_ex)
        # 记录刚学完本批时的峰值成绩
        m_now = eval_batch(model, store, graph, b["eval"], abstain_thresh=0.5, budget=budget)
        peak_scores[b["id"]] = m_now["score"]
        peak_cross[b["id"]] = m_now["cross_exact"]

    # 全部 5 批学完后，重测全部 5 批
    final_evals = {
        b["id"]: eval_batch(model, store, graph, b["eval"], abstain_thresh=0.5, budget=budget)
        for b in batches
    }
    old_ids = [b["id"] for b in batches[:-1]]  # Batches 1..4
    old_final_scores = [final_evals[i]["score"] for i in old_ids]
    old_drops = [peak_scores[i] - final_evals[i]["score"] for i in old_ids]
    old_cross_drops = [peak_cross[i] - final_evals[i]["cross_exact"] for i in old_ids]

    return {
        "active_params": count_active_params(model),
        "new_batch_score": final_evals[batches[-1]["id"]]["score"],
        "new_batch_cross": final_evals[batches[-1]["id"]]["cross_exact"],
        "mean_old_score": round(sum(old_final_scores) / len(old_final_scores), 4),
        "worst_batch_score": round(min(old_final_scores), 4),
        "mean_old_drop": round(sum(old_drops) / len(old_drops), 4),
        "worst_batch_drop": round(max(old_drops), 4),
        "worst_cross_drop": round(max(old_cross_drops), 4),
        "mean_fpr": round(sum(final_evals[b["id"]]["hard_neg_fpr"] for b in batches) / len(batches), 4),
        "per_batch_final": final_evals,
        "per_batch_peak": peak_scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_per_batch", type=int, default=28)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== E3 2×2 严格消融（共享网络 vs RMP × 有回放 vs 无回放，seeds={seeds}）===", flush=True)

    encoder = FrozenTextEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    store, graph, batches = build_e3_five_batches(encoder)

    grid = [
        ("shared_no_replay", "shared", False),
        ("shared_with_replay", "shared", True),
        ("rmp_no_replay", "rmp", False),
        ("rmp_with_replay", "rmp", True),
    ]

    all_runs: Dict[str, List[Dict]] = {name: [] for name, _, _ in grid}
    for name, arch, rep in grid:
        for sd in seeds:
            r = run_condition(arch, rep, store, graph, batches, args.epochs_per_batch, args.lr, args.budget, sd, dev)
            all_runs[name].append(r)
        print(f"完成 {name}（active_params={all_runs[name][0]['active_params']}）", flush=True)

    summary = {}
    print(f"\n=== E3 2×2 消融汇总（{len(seeds)} 种子均值，重点指标：Worst-Batch Retention / Drop）===")
    header = f"{'条件':<20} | {'活跃参':>6} | {'新批得分':>7} | {'旧批均分':>7} | {'最差旧批↑':>8} | {'最差退步↓':>8} | {'最差跨事件掉↓':>10} | {'均值FPR↓':>7}"
    print(header)
    print("-" * len(header))
    for name, _, _ in grid:
        rs = all_runs[name]
        s = {
            "active_params": rs[0]["active_params"],
            "new_batch_score": round(sum(r["new_batch_score"] for r in rs) / len(rs), 4),
            "mean_old_score": round(sum(r["mean_old_score"] for r in rs) / len(rs), 4),
            "worst_batch_score": round(sum(r["worst_batch_score"] for r in rs) / len(rs), 4),
            "mean_old_drop": round(sum(r["mean_old_drop"] for r in rs) / len(rs), 4),
            "worst_batch_drop": round(sum(r["worst_batch_drop"] for r in rs) / len(rs), 4),
            "worst_cross_drop": round(sum(r["worst_cross_drop"] for r in rs) / len(rs), 4),
            "mean_fpr": round(sum(r["mean_fpr"] for r in rs) / len(rs), 4),
        }
        summary[name] = s
        print(
            f"{name:<20} | {s['active_params']:>6d} | {s['new_batch_score']:>7.1%} | "
            f"{s['mean_old_score']:>7.1%} | {s['worst_batch_score']:>8.1%} | "
            f"{s['worst_batch_drop']:>+8.1%} | {s['worst_cross_drop']:>+10.1%} | {s['mean_fpr']:>7.1%}",
            flush=True,
        )

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_e3.json"
    payload = {
        "experiment": "E3_rmp_2x2_ablation",
        "encoder": "BAAI/bge-small-zh-v1.5",
        "budget": args.budget,
        "epochs_per_batch": args.epochs_per_batch,
        "seeds": seeds,
        "elapsed_s": elapsed,
        "summary_mean": summary,
        "all_runs": all_runs,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
