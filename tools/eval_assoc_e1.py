"""E1 实验：验证可学习循环回想（含跳出图全库重搜 + 缺证/拒答门控）是否有真实增益。

严格遵循 docs/design/associative-memory.md 与三项升级：
1. 训练与评测严格按「主题 + 实体链」隔离（训练集使用 12 个完全无关的日常主题，评测集使用 24 篇阿远自然日记）。
2. 关闭 LLM，纯评事件 ID。
3. 相同事件库、相同冻结编码器（BAAI/bge-small-zh-v1.5）、相同搜索预算（budget=16），公平对比：
   - one_shot（一次语义检索，训练集校准阈值）
   - fixed_2hop_graph（固定图两跳传播，训练集校准阈值）
   - plain_2round（普通两轮检索：首轮结果固定改写查询重搜全库，训练集校准阈值）
   - learned_hybrid（可学习循环回想：图邻居 + 跳出图全库重搜，主版）
   - learned_graph_only（消融：可学习循环回想，仅图邻居）
   - learned_full_only（消融：可学习循环回想，仅全库重搜）
4. 因果消融：移掉桥接事件（drop_bridge）vs 移掉无关事件（drop_unrelated）。
5. 原始结果保存为 runs/<timestamp>_assoc_e1.json。
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
    Fixed2HopGraphBaseline,
    FrozenTextEncoder,
    IterativeRecallNet,
    OneShotBaseline,
    PlainTwoRoundBaseline,
    RecallExample,
    RecallResult,
    SparseAssociativeGraph,
    train_iterative_recall,
)
from tools.test_diary_10k import NO_KEYWORD_PROMPTS, build_10k_diary


# ---------------------------------------------------------------------------
# 1. 训练集（12 个与阿远日记完全无交集的日常主题与实体链，零泄露）
# ---------------------------------------------------------------------------

def build_disjoint_train_world(encoder: FrozenTextEncoder) -> Tuple[EpisodicStore, SparseAssociativeGraph, List[RecallExample]]:
    store = EpisodicStore(encoder)
    events = [
        # 主题 1：橘猫团子 & 寄养/喂食
        ("t_cat_1", 1, "tr_01", "家里那只胖橘猫团子胆子很小，一听见楼道装修电钻声就会躲进衣柜最里层不肯出来。"),
        ("t_cat_2", 14, "tr_14", "出差前把团子送去了巷口的暖窝宠物寄养店，店主小周说每天会发视频报平安。"),
        ("t_cat_3", 22, "tr_22", "给团子买的自动喂食器卡粮了，回家发现碗里空空如也它正蹲在旁边瞪我。"),
        # 主题 2：民谣吉他 & 琴行老赵
        ("t_gui_1", 2, "tr_02", "练指弹曲《晴天》时二弦突然崩断，打在手指上火辣辣地疼，只能先暂停练习。"),
        ("t_gui_2", 16, "tr_16", "把民谣吉他送去星海琴行找老赵师傅换了一套达达里奥琴弦并顺便调了琴颈弦距。"),
        # 主题 3：慢跑 & 膝盖理疗
        ("t_run_1", 3, "tr_03", "昨晚绕东湖跑了十公里，最后两公里右膝外侧开始隐隐作痛，走楼梯都发软。"),
        ("t_run_2", 18, "tr_18", "去仁济康复中心找林医生做右膝理疗，他叮嘱未来两周只能做静蹲不能长跑。"),
        # 主题 4：阳台多肉 & 邻居小孙
        ("t_plt_1", 4, "tr_04", "阳台那盆熊童子多肉最近叶片发软起皱，查了资料发现是连续暴晒缺水导致的。"),
        ("t_plt_2", 19, "tr_19", "国庆回老家五天，把阳台花洒和门禁卡交给对门邻居小孙，拜托他隔两天给多肉喷点水。"),
        # 主题 5：公路车 & 车店阿强
        ("t_bik_1", 5, "tr_05", "早高峰骑公路车通勤经过高架桥下时，后变速器咔哒咔哒异响，差点掉链子。"),
        ("t_bik_2", 20, "tr_20", "周末推着公路车去风行车店找阿强师傅拆洗飞轮并重新上了干性链条油。"),
        # 主题 6：母亲生日 & 快递自提
        ("t_mom_1", 6, "tr_06", "下周三是妈妈六十岁生日，挑了很久最后选了一条暗红色的鄂尔多斯羊绒围巾。"),
        ("t_mom_2", 21, "tr_21", "顺丰小哥把羊绒围巾放进了小区南门丰巢快递柜三号箱，取件码发到了短信里。"),
        # 主题 7：酸种面包 & 烤箱温度
        ("t_bak_1", 7, "tr_07", "周末试烤欧式酸种面包，出炉发现底部焦黑里面还没熟透，怀疑是烤箱温控偏高。"),
        ("t_bak_2", 23, "tr_23", "买了一个机械式炉内温度计挂在烤箱中层，实测设定两百度时炉内其实高达两百三十度。"),
        # 主题 8：牙科复诊（含已变更）
        ("t_den_old", 8, "tr_08", "原本预约了周四上午十点去明德口腔找王大夫做正畸复诊换弓丝。"),
        ("t_den_new", 24, "tr_24", "明德口腔护士打电话通知，王大夫周四外出开会，我的正畸复诊改到了周六下午三点。"),
        # 主题 9：日语 N2 备考 & 图书馆
        ("t_jpn_1", 9, "tr_09", "做完一套日语 N2 模拟卷，发现长篇阅读和听力即时应答错得最多，心里有点慌。"),
        ("t_jpn_2", 25, "tr_25", "办了市图书馆三楼静音自习室的座位卡，准备每晚下班去那儿刷两小时 N2 听力真题。"),
        # 主题 10：露营装备 & 同事大伟
        ("t_cmp_1", 10, "tr_10", "计划月底去莫干山竹林营地过夜，晚上清点装备发现防潮垫和营地灯都不见了。"),
        ("t_cmp_2", 26, "tr_26", "翻聊天记录才想起来，上个月把防潮垫和营地灯借给财务部同事大伟去千岛湖用了还没还。"),
        # 主题 11：笔记本风扇 & 清灰
        ("t_lap_1", 11, "tr_11", "家里那台老款轻薄本一开视频会议风扇就狂转像飞机起飞，键盘左侧烫得没法放手。"),
        ("t_lap_2", 27, "tr_27", "把发烫的轻薄本带去百脑汇三楼维修摊清了散热模组灰尘并更换了霍尼韦尔硅脂。"),
        # 主题 12：游泳馆（含已变更）
        ("t_swm_old", 12, "tr_12", "平时周二晚上都会去区体育中心室内恒温游泳馆游一千米蛙泳。"),
        ("t_swm_new", 28, "tr_28", "区体育中心泳池闭馆检修一个月，这段时间改去河滨健身俱乐部二楼泳池游泳。"),
        # 主题 13：热水器报修 & 房东李姐
        ("t_wtr_1", 13, "tr_31", "昨晚洗澡到一半燃气热水器突然报 E4 故障代码直接熄火，差点冻感冒。"),
        ("t_wtr_2", 29, "tr_32", "联系房东李姐后，她约了万和售后张师傅周六上午十点上门免费更换风压开关。"),
        # 主题 14：立式钢琴 & 调音师陈师傅
        ("t_pno_1", 14, "tr_33", "梅雨季过后客厅那台立式钢琴的高音区好几个白键按下去回弹变慢还带杂音。"),
        ("t_pno_2", 30, "tr_34", "加了音乐学院推荐的调音师陈师傅微信，约好下周二带除湿棒上门给钢琴做整音调律。"),
        # 主题 15：年度体检 & 控糖饮食
        ("t_chk_1", 15, "tr_35", "拿到公司年度体检报告，空腹血糖和甘油三酯双双偏高，内科大夫让我少吃精制碳水。"),
        ("t_chk_2", 31, "tr_36", "照着内科大夫的建议买了十斤三色糙米和燕麦米，把家里电饭煲晚饭主食全换成了杂粮饭。"),
        # 主题 16：净水器滤芯 & 社区团购
        ("t_flt_1", 16, "tr_37", "厨房水槽下的反渗透净水器面板亮红灯报警，算算时间 RO 膜滤芯已经超期两个月了。"),
        ("t_flt_2", 32, "tr_38", "在小区业主群里跟着团长拼单买了两根原厂 RO 膜滤芯，明天晚上去西门驿站拿货自己拧上。"),
        # 主题 17：耳机遗落 & 失物招领
        ("t_ear_1", 17, "tr_39", "周五晚上打网约车回家太困，下车时把降噪蓝牙耳机充电盒落在了后排座椅缝里。"),
        ("t_ear_2", 33, "tr_40", "网约车司机刘师傅捡到降噪耳机后帮我寄存在了公司楼下门卫室，报手机尾号 6842 就能领。"),
        # 主题 18：驾照换证 & 车管所
        ("t_drv_1", 18, "tr_41", "整理钱包发现六年期的 C1 驾驶证还有二十天就到期了，必须尽快办期满换证。"),
        ("t_drv_2", 34, "tr_42", "在交管 12123 查到城东自助体检机周末也开门，打算周日上午去拍照体检顺便现场制新证。"),
        # 近重复干扰事件（仅在独立训练世界，教模型第二跳过滤近重复复读机、选择真正补充缺失信息的目标）
        ("t_cat_dup", 3, "tr_03b", "楼道里一响起电钻声，胖橘猫团子就会吓得钻进衣柜深处发抖半天不敢露头。"),
        ("t_run_dup", 5, "tr_05b", "前几天去东湖长跑完右膝外侧也隐隐发疼，下楼梯时膝盖发软不太敢用力。"),
        ("t_lap_dup", 12, "tr_12b", "老轻薄本每次开视频会议风扇都嗡嗡狂转，键盘左边烫手得几乎没法打字。"),
        # 额外背景干扰事件（增加事件库密度）
        ("t_bg_1", 13, "tr_13", "午休在公司茶水间泡了一杯龙井茶，和实习生聊了聊实习报告的排版格式。"),
        ("t_bg_2", 15, "tr_15", "今天地铁二号线早高峰信号故障延误了二十分钟，到公司刚好卡点打卡。"),
        ("t_bg_3", 17, "tr_17", "下班路过菜市场买了两斤番茄和一把小葱，回家煮了碗热腾腾的番茄鸡蛋面。"),
        ("t_bg_4", 29, "tr_29", "整理书桌抽屉翻出一张前年去西安旅游的兵马俑门票，夹进了手账本里。"),
        ("t_bg_5", 30, "tr_30", "物业群通知周日上午九点到十一点停水清洗水箱，提前接了两桶备用自来水。"),
    ]
    for eid, day, src, txt in events:
        if eid == "t_den_new":
            store.supersede_event("t_den_old", txt, day=day, source=src, new_id=eid)
        elif eid == "t_swm_new":
            store.supersede_event("t_swm_old", txt, day=day, source=src, new_id=eid)
        else:
            store.add_event(txt, day=day, source=src, event_id=eid)

    graph = SparseAssociativeGraph(store, max_neighbors=16)

    examples = [
        # --- 1. 跨事件补全（省去桥接实体名）---
        RecallExample("楼道又开始电钻装修了，我出差不在家，躲进衣柜的小家伙现在安顿在哪儿？", "cross", ["t_cat_1", "t_cat_2"], "cat", "t_cat_1", "t_cat_2"),
        RecallExample("弹《晴天》崩断弦的那把乐器，后来送去哪儿找谁修好了？", "cross", ["t_gui_1", "t_gui_2"], "guitar", "t_gui_1", "t_gui_2"),
        RecallExample("跑完十公里腿疼发软，医生让接下来两周怎么恢复？", "cross", ["t_run_1", "t_run_2"], "run", "t_run_1", "t_run_2"),
        RecallExample("国庆回老家那几天，阳台怕暴晒缺水的盆栽托付给谁照看了？", "cross", ["t_plt_1", "t_plt_2"], "plant", "t_plt_1", "t_plt_2"),
        RecallExample("早高峰骑车变速异响差点掉链子，周末推去哪儿保养的？", "cross", ["t_bik_1", "t_bik_2"], "bike", "t_bik_1", "t_bik_2"),
        RecallExample("给妈妈六十岁生日挑的那份暗红色礼物，快递放进哪个自提柜了？", "cross", ["t_mom_1", "t_mom_2"], "mom", "t_mom_1", "t_mom_2"),
        RecallExample("烤欧式面包底部焦黑怀疑温控不准，后来测出炉内到底高了多少度？", "cross", ["t_bak_1", "t_bak_2"], "bake", "t_bak_1", "t_bak_2"),
        RecallExample("模拟卷听力和阅读错得多心里发慌，我打算晚上去哪儿刷真题？", "cross", ["t_jpn_1", "t_jpn_2"], "jpn", "t_jpn_1", "t_jpn_2"),
        RecallExample("月底去莫干山过夜清点装备发现少了两样，原来借给谁还没拿回来？", "cross", ["t_cmp_1", "t_cmp_2"], "camp", "t_cmp_1", "t_cmp_2"),
        RecallExample("开视频会议键盘烫得没法放手的那台电脑，后来去哪儿清灰换硅脂了？", "cross", ["t_lap_1", "t_lap_2"], "laptop", "t_lap_1", "t_lap_2"),
        # 再增加不同问法（增强泛化）
        RecallExample("家里装修电钻吵得不行，平时胆小爱躲衣柜的宠物是不是送去小周那儿了？", "cross", ["t_cat_1", "t_cat_2"], "cat", "t_cat_1", "t_cat_2"),
        RecallExample("东湖跑完步走楼梯发软，康复中心的林医生叮嘱了什么？", "cross", ["t_run_1", "t_run_2"], "run", "t_run_1", "t_run_2"),
        RecallExample("月底露营装备找不齐，是借给财务部哪位同事去千岛湖了？", "cross", ["t_cmp_1", "t_cmp_2"], "camp", "t_cmp_1", "t_cmp_2"),
        RecallExample("老电脑风扇像飞机起飞一样狂转，修完换的是什么牌子的硅脂？", "cross", ["t_lap_1", "t_lap_2"], "laptop", "t_lap_1", "t_lap_2"),
        RecallExample("昨晚洗澡报 E4 熄火差点冻感冒，房东约了哪位师傅周六几点上门修？", "cross", ["t_wtr_1", "t_wtr_2"], "water", "t_wtr_1", "t_wtr_2"),
        RecallExample("梅雨季后高音区白键回弹变慢带杂音，约了哪位老师下周二带什么上门处理？", "cross", ["t_pno_1", "t_pno_2"], "piano", "t_pno_1", "t_pno_2"),
        RecallExample("年度体检报告显示指标偏高大夫让少吃精制碳水，后来把晚饭主食换成了什么？", "cross", ["t_chk_1", "t_chk_2"], "checkup", "t_chk_1", "t_chk_2"),
        RecallExample("厨房水槽下面板亮红灯超期两个月了，拼单买的新配件明天去哪儿取货？", "cross", ["t_flt_1", "t_flt_2"], "filter", "t_flt_1", "t_flt_2"),
        RecallExample("周五打车太困落在后排座椅缝里的小物件，司机师傅寄存在哪儿、报什么尾号去领？", "cross", ["t_ear_1", "t_ear_2"], "earbud", "t_ear_1", "t_ear_2"),
        RecallExample("整理钱包发现六年期的证件还有二十天到期，周末打算去城东哪儿现场办理？", "cross", ["t_drv_1", "t_drv_2"], "driver", "t_drv_1", "t_drv_2"),

        # --- 2. 事件补全（单跳直接回想）---
        RecallExample("给橘猫团子买的自动喂食器出了什么故障？", "direct", ["t_cat_3"], "cat"),
        RecallExample("周日小区停水清洗水箱是几点到几点？", "direct", ["t_bg_5"], "bg"),
        RecallExample("地铁二号线早高峰为什么延误了二十分钟？", "direct", ["t_bg_2"], "bg"),
        RecallExample("整理书桌抽屉翻出了哪次旅游的门票？", "direct", ["t_bg_4"], "bg"),
        RecallExample("下班路过菜市场买菜回家煮了什么面？", "direct", ["t_bg_3"], "bg"),
        RecallExample("午休在茶水间泡了什么茶和实习生聊天？", "direct", ["t_bg_1"], "bg"),

        # --- 3. 缺证但仅存半条（partial_only：只停在已知事实，不乱拉第二跳）---
        RecallExample("自动喂食器卡粮那次，后来有没有换新品牌的猫粮？", "partial_only", ["t_cat_3"], "cat"),
        RecallExample("整理抽屉翻出西安兵马俑门票，当时住的是哪家酒店？", "partial_only", ["t_bg_4"], "bg"),

        # --- 4. 已变更（superseded：必须返回 active 新安排，绝不返回旧值）---
        RecallExample("我预约的明德口腔正畸复诊现在具体是哪天几点？", "superseded", ["t_den_new"], "dentist"),
        RecallExample("最近周二晚上想去游蛙泳，现在应该去哪个泳池？", "superseded", ["t_swm_new"], "swim"),

        # --- 5. 相似但无关难负例（hard_neg：必须拒答）---
        RecallExample("橘猫为什么冬天容易发胖，每天喂多少克猫粮合适？", "hard_neg", [], "neg"),
        RecallExample("民谣吉他新手换弦时怎么防止二弦崩断？", "hard_neg", [], "neg"),
        RecallExample("跑步膝盖外侧髂胫束综合征怎么拉伸缓解？", "hard_neg", [], "neg"),
        RecallExample("熊童子多肉夏天休眠期多久浇一次水不会烂根？", "hard_neg", [], "neg"),
        RecallExample("公路车链条油选干性还是湿性比较好？", "hard_neg", [], "neg"),
        RecallExample("羊绒围巾手洗缩水了怎么恢复原状？", "hard_neg", [], "neg"),
        RecallExample("欧式酸种面包天然酵种怎么从零开始培养？", "hard_neg", [], "neg"),
        RecallExample("日语 N2 考试每年几月份报名，及格线是多少分？", "hard_neg", [], "neg"),
        RecallExample("莫干山周末自驾游有哪些性价比高的民宿推荐？", "hard_neg", [], "neg"),
        RecallExample("笔记本电脑涂硅脂用九点法还是刮平法散热更好？", "hard_neg", [], "neg"),
        RecallExample("蛙泳换气总是呛水有什么呼吸技巧？", "hard_neg", [], "neg"),
        RecallExample("顺丰丰巢快递柜超时存放超过 18 小时怎么收费？", "hard_neg", [], "neg"),
        RecallExample("燃气热水器显示 E4 故障代码自己在家怎么重启复位？", "hard_neg", [], "neg"),
        RecallExample("立式钢琴一年需要调音几次，室内湿度保持在多少合适？", "hard_neg", [], "neg"),
        RecallExample("甘油三酯偏高人群日常饮食应该注意哪些禁忌？", "hard_neg", [], "neg"),
        RecallExample("反渗透 RO 膜净水器滤芯一般几年更换一次性价比最高？", "hard_neg", [], "neg"),
        RecallExample("C1 驾驶证期满换证在网上交管 12123 申请需要上传什么照片？", "hard_neg", [], "neg"),

        # --- 6. 证据不足（insufficient：库中无记录，应拒答）---
        RecallExample("我去年考驾照科目二是在哪个考场通过的？", "insufficient", [], "insuff"),
        RecallExample("家里客厅空调遥控器上次换的是几号电池？", "insufficient", [], "insuff"),
        RecallExample("我大学毕业论文的指导老师姓什么？", "insufficient", [], "insuff"),
        RecallExample("上个月信用卡账单还款日是每个月几号？", "insufficient", [], "insuff"),
    ]
    return store, graph, examples


# ---------------------------------------------------------------------------
# 2. 评测集（基于 tools/test_diary_10k.py 的 24 篇阿远自然日记切分 + 独立 4 类评测题 + 原 6 题冒烟）
# ---------------------------------------------------------------------------

def _parse_diary_day(header: str, idx: int) -> int:
    m = re.search(r"(\d+)月(\d+)日", header)
    if not m:
        return idx + 1
    month, day = int(m.group(1)), int(m.group(2))
    return (month - 10) * 31 + day


def build_ayuan_eval_world(
    encoder: FrozenTextEncoder,
) -> Tuple[EpisodicStore, SparseAssociativeGraph, Dict[str, List[RecallExample]], List[Tuple[str, str, List[str]]]]:
    """把 24 篇自然日记切分为带来源、时间戳、事件 ID 的事件库，并加入真实后续更新/覆盖事件。"""
    entries, _ = build_10k_diary()
    store = EpisodicStore(encoder)

    # 先把 24 篇自然日记按句切分为原子事件（去重完全重复的句段，保留首次出现日期与来源；
    # 保证 24 篇全部纳入事件库且每条核心经历拥有清晰唯一的事件 ID）
    seen_texts = set()
    ev_counter = 0
    for d_idx, raw in enumerate(entries):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        header = lines[0] if lines else f"diary_{d_idx + 1:02d}"
        day = _parse_diary_day(header, d_idx)
        src = f"diary_{d_idx + 1:02d}"
        body_lines = lines[1:]
        for para in body_lines:
            # 按中文句号/感叹号切分为原子事件（每条 25~120 字）
            sents = [s.strip() for s in re.split(r"(?<=[。！？])", para) if s.strip()]
            buf = ""
            for s in sents:
                if len(buf) + len(s) < 28:
                    buf += s
                    continue
                cand = (buf + s) if buf else s
                buf = ""
                if cand in seen_texts:
                    continue
                seen_texts.add(cand)
                ev_counter += 1
                store.add_event(cand, day=day, source=src, event_id=f"d_{ev_counter:03d}")
            if buf and buf not in seen_texts:
                seen_texts.add(buf)
                ev_counter += 1
                store.add_event(buf, day=day, source=src, event_id=f"d_{ev_counter:03d}")

    # 额外加入 4 条真实的「后续新增经历 / 计划变更」（均遵循自然日记风格，用于测试：
    # 1) 跳出图找最新照看安排；2) 旧安排 superseded 被新安排覆盖）
    # a) 新增照看安排（无雷雨字眼，仅通过「旺财 / 南岸公寓」与旧经历关联）
    e_care_old = store.add_event(
        "原本打算加班晚归时把南岸公寓的备用钥匙暂存在一楼物业前台。",
        day=35,
        source="diary_25_upd",
        event_id="u_care_old",
    )
    e_care_new = store.supersede_event(
        e_care_old.event_id,
        "物业前台晚上八点就下班了，最后改把南岸公寓的备用钥匙交给了隔壁邻居陈阿姨，托她我加班晚归时帮忙开门照看旺财。",
        day=48,
        source="diary_26_upd",
        new_id="u_care_new",
    )
    # b) 羽毛球时间变更（旧周三 -> 新周五）
    e_badm_old = store.add_event(
        "十月初和前端组同事约好固定每周三晚上七点去体育馆打羽毛球。",
        day=3,
        source="diary_02_sched",
        event_id="u_badm_old",
    )
    e_badm_new = store.supersede_event(
        e_badm_old.event_id,
        "体育馆周三场地被学校包场了，前端组羽毛球局从十一月起改到了每周五晚上八点。",
        day=42,
        source="diary_27_upd",
        new_id="u_badm_new",
    )
    # c) 明年年假目的地变更（旧冲绳 -> 新北海道）
    e_trip_old = store.add_event(
        "年初原本计划明年休年假去冲绳潜水看海，还收藏了几家海边民宿。",
        day=2,
        source="diary_01_plan",
        event_id="u_trip_old",
    )
    e_trip_new = store.supersede_event(
        e_trip_old.event_id,
        "改主意不去冲绳了，决定明年冬天休年假去北海道看雪，顺便背着老富士胶片相机多拍几卷雪景胶片。",
        day=45,
        source="diary_28_upd",
        new_id="u_trip_new",
    )
    # d) 咖啡口味变更（旧全脂热拿铁 -> 新少冰燕麦拿铁）
    e_cof_old = store.add_event(
        "刚搬来南岸时早上习惯在楼下咖啡馆点全脂牛奶热拿铁。",
        day=1,
        source="diary_01_cof",
        event_id="u_cof_old",
    )
    e_cof_new = store.supersede_event(
        e_cof_old.event_id,
        "最近乳糖不耐受改喝植物奶了，在南岸楼下咖啡馆固定点少冰燕麦拿铁，店员不用我开口都记得。",
        day=44,
        source="diary_29_upd",
        new_id="u_cof_new",
    )
    # e) 冲扫店信息（与老富士胶片相机形成无直接海边/旅行字眼的跨事件桥接）
    e_film_shop = store.add_event(
        "市中心弗林德斯街那家黑白暗房冲扫店手艺很稳，老富士拍完的胶卷我一般都送去那儿冲扫。",
        day=50,
        source="diary_30_upd",
        event_id="u_film_shop",
    )
    # f) 医生过敏提醒（与海鲜过敏事件桥接）
    e_med_kit = store.add_event(
        "家庭医生开的氯雷他定抗过敏药片一直放在玄关抽屉最上层，误食虾蟹起疹子时救急用。",
        day=52,
        source="diary_31_upd",
        event_id="u_med_kit",
    )

    graph = SparseAssociativeGraph(store, max_neighbors=16)

    # 通过原文精准定位 24 篇日记切分出的事件 ID（不硬编码序号，防止切分微调错位）
    def find_id(substr: str) -> str:
        for ev in store.active_events():
            if substr in ev.text:
                return ev.event_id
        raise KeyError(f"未在事件库找到包含 {substr!r} 的事件")

    e_self = find_id("我是林远，朋友们平时都叫我阿远")
    e_southbank = find_id("住在墨尔本南岸")
    e_wangcai_park = find_id("三岁的金毛犬旺财去阿尔伯特公园")
    e_coke = find_id("顺路在便利店拎了两罐冰镇无糖可乐")
    e_dinner = find_id("自己煎了两块鸡胸肉配水煮西兰花")
    e_fe_dev = find_id("天天跟 TypeScript 和 React 打交道的前端开发")
    e_allergy_refuse = find_id("同事约着去吃海鲜大排档，我赶紧摆手拒绝了")
    e_allergy_child = find_id("小时候不懂事偷吃过一次大虾")
    e_thunder_desk = find_id("一听见打雷就缩到我书桌底下发抖")
    e_thunder_night = find_id("半夜两点突然电闪雷鸣")
    e_thunder_work = find_id("每次外面乌云密布要打雷，我就算在公司加班")
    e_fuji_clean = find_id("老富士胶片相机拿出来擦了擦镜头")
    e_santi = find_id("窝在沙发里重读《三体》")
    e_galaxy = find_id("翻了五十页阿西莫夫的《银河帝国》")
    e_app_proto = find_id("画了画我一直想做的那个独立记账 App 的原型图")
    e_app_burnout = find_id("今年二十八岁了，天天在公司改 React 组件")
    e_app_advice = find_id("同事劝我先利用周末把原型跑通")
    e_app_code = find_id("写了两个小时独立记账 App 的前端框架")
    e_hokkaido_budget = find_id("把明年的北海道看雪预算粗略列了个表")
    e_ts_bug = find_id("诡异的 TypeScript 异步状态竞争 Bug")
    e_photo_wall = find_id("有一张旺财在夕阳下叼着网球奔跑的抓拍")

    # 4 类独立评测题（全部人工核验，与训练集零主题/零实体交集）
    eval_sets: Dict[str, List[RecallExample]] = {
        "direct": [
            RecallExample("朋友们平时都怎么称呼我，我全名叫什么？", "direct", [e_self], "self"),
            RecallExample("住在墨尔本南岸走几步就能看到哪条河？", "direct", [e_southbank], "loc"),
            RecallExample("冰箱里要是不囤够什么饮料我心里就不踏实？", "direct", [e_coke], "drink"),
            RecallExample("平时懒得折腾晚饭一般自己做什么菜？", "direct", [e_dinner], "food"),
            RecallExample("我在公司日常开发主要用哪两个前端技术栈？", "direct", [e_fe_dev], "work"),
            RecallExample("上次在公司从上午十点查到下午四点修的是什么 Bug？", "direct", [e_ts_bug], "work"),
            RecallExample("晚上窝在沙发里重读哪本科幻小说读到黑暗森林觉得头皮发麻？", "direct", [e_santi], "book"),
            RecallExample("睡前翻了五十页阿西莫夫的哪本小说来忘掉写代码的烦心事？", "direct", [e_galaxy], "book"),
            RecallExample("那卷海边胶片里哪张抓拍我觉得可以直接挂客厅墙上？", "direct", [e_photo_wall], "photo"),
            RecallExample("平时写完晚饭不刷短视频，打开电脑写了两个小时什么项目的框架？", "direct", [e_app_code], "app"),
        ],
        # 跨事件回想：省去桥接实体名（不提 旺财 / 海鲜 / 富士 / 独立记账 App / 北海道 等桥接词）
        "cross": [
            RecallExample(
                "今晚外面打雷下大雨我还在加班回不去，家里那个怕雷的家伙能找谁开门帮忙照看？",
                "cross", [e_thunder_desk, e_care_new.event_id], "pet_thunder", e_thunder_desk, e_care_new.event_id,
            ),
            RecallExample(
                "半夜电闪雷鸣吓得钻床边那个小家伙，平时周末下午最喜欢去哪个公园追网球？",
                "cross", [e_thunder_night, e_wangcai_park], "pet_park", e_thunder_night, e_wangcai_park,
            ),
            RecallExample(
                "在公司加班看窗外乌云密布惦记着家里缩桌底发抖的小家伙，备用钥匙现在放在谁那儿？",
                "cross", [e_thunder_work, e_care_new.event_id], "pet_key", e_thunder_work, e_care_new.event_id,
            ),
            RecallExample(
                "打完球同事约去吃大排档我赶紧摆手拒绝了，要是真不小心误食起疹子，急救药片放在家里哪儿？",
                "cross", [e_allergy_refuse, e_med_kit.event_id], "allergy_med", e_allergy_refuse, e_med_kit.event_id,
            ),
            RecallExample(
                "小时候偷吃过一次差点进医院挂水，所以现在跟同事聚餐拒绝那家大排档后，我晚饭一般吃什么？",
                "cross", [e_allergy_child, e_dinner], "allergy_dinner", e_allergy_child, e_dinner,
            ),
            RecallExample(
                "下雨没法出门在家擦镜头整理圣基尔达海边那卷照片，以后拍完卷一般送去市中心哪家店冲扫？",
                "cross", [e_fuji_clean, e_film_shop.event_id], "camera_shop", e_fuji_clean, e_film_shop.event_id,
            ),
            RecallExample(
                "下雨在家擦镜头整理海边拍的胶片，明年冬天休年假打算背着它去哪儿拍雪景？",
                "cross", [e_fuji_clean, e_trip_new.event_id], "camera_trip", e_fuji_clean, e_trip_new.event_id,
            ),
            RecallExample(
                "天天在公司改业务表单少了掌控感、想自己做点喜欢的小产品，同事给我的建议是什么？",
                "cross", [e_app_burnout, e_app_advice], "app_advice", e_app_burnout, e_app_advice,
            ),
            RecallExample(
                "周末在笔记本上画了极简原型图的那个小项目，最近下班后花两小时开始写什么了？",
                "cross", [e_app_proto, e_app_code], "app_code", e_app_proto, e_app_code,
            ),
            RecallExample(
                "同事劝我先攒够旅行基金和一年生活费再决定辞职，那趟冬日看雪行程的预算评估过能不能实现吗？",
                "cross", [e_app_advice, e_hokkaido_budget], "budget_trip", e_app_advice, e_hokkaido_budget,
            ),
            RecallExample(
                "海边夕阳下叼着网球奔跑那张抓拍的主角，一遇到雷雨天就会躲到哪里发抖？",
                "cross", [e_photo_wall, e_thunder_desk], "photo_thunder", e_photo_wall, e_thunder_desk,
            ),
            RecallExample(
                "天天写代码坐办公室腰酸背痛靠打球挥拍流汗，现在前端组这项运动改到周几几点了？",
                "cross", [e_fe_dev, e_badm_new.event_id], "work_badm", e_fe_dev, e_badm_new.event_id,
            ),
        ],
        # 相似但无关难负例（语义极近，必须拒答）
        "hard_neg": [
            RecallExample("拍闪电和雷暴天气的相机快门速度和感光度参数应该怎么设置？", "hard_neg", [], "neg"),
            RecallExample("墨尔本市中心有哪些评分比较高的海鲜自助餐厅推荐？", "hard_neg", [], "neg"),
            RecallExample("React 19 新的 useActionState 钩子怎么配合异步表单使用？", "hard_neg", [], "neg"),
            RecallExample("北海道二世古滑雪场的早鸟缆车票怎么在官网预订？", "hard_neg", [], "neg"),
            RecallExample("金毛犬春秋换毛季掉毛严重，推荐什么牌子的针梳？", "hard_neg", [], "neg"),
            RecallExample("《三体》动画版和电视剧版在黑暗森林剧情上差别大吗？", "hard_neg", [], "neg"),
            RecallExample("羽毛球拍拉线磅数选 24 磅还是 26 磅更适合业余新手？", "hard_neg", [], "neg"),
            RecallExample("鸡胸肉怎么腌制才能煎出来鲜嫩多汁不发柴？", "hard_neg", [], "neg"),
            RecallExample("雅拉河游船晚宴的登船码头具体在哪个街口？", "hard_neg", [], "neg"),
            RecallExample("富士数码相机 X-T5 和 X100VI 拍人像胶片模拟有什么区别？", "hard_neg", [], "neg"),
            RecallExample("燕麦奶和普通全脂牛奶做拿铁拉花哪个更容易出绵密奶泡？", "hard_neg", [], "neg"),
            RecallExample("阿西莫夫《银河帝国》基地系列正确的出版阅读顺序是什么？", "hard_neg", [], "neg"),
        ],
        # 已变更（必须返回最新 active 事件，绝不返回旧 superseded 事件）
        "superseded": [
            RecallExample("前端组的羽毛球局现在改到每周几晚上几点打了？", "superseded", [e_badm_new.event_id], "sched"),
            RecallExample("明年年假最终决定去哪儿玩，还去冲绳吗？", "superseded", [e_trip_new.event_id], "trip"),
            RecallExample("现在在南岸楼下咖啡馆早上固定点什么咖啡？", "superseded", [e_cof_new.event_id], "coffee"),
            RecallExample("南岸公寓的备用钥匙现在到底放在物业前台还是谁那儿？", "superseded", [e_care_new.event_id], "key"),
        ],
        # 证据不足（库中真没有答案，应拒答而不是乱联想）
        "insufficient": [
            RecallExample("旺财上次打狂犬疫苗和驱虫是在哪家宠物医院做的？", "insufficient", [], "insuff"),
            RecallExample("老富士胶片相机配的那颗镜头具体是多少毫米焦段、多大光圈？", "insufficient", [], "insuff"),
            RecallExample("独立记账 App 计划定在今年几月几号正式上线发布？", "insufficient", [], "insuff"),
            RecallExample("南岸这套公寓现在的每月房租和押金一共是多少澳元？", "insufficient", [], "insuff"),
            RecallExample("明年去北海道看雪已经订好哪家航空公司的往返航班号了？", "insufficient", [], "insuff"),
            RecallExample("前端组打羽毛球每次场地费 AA 下来人均多少钱？", "insufficient", [], "insuff"),
        ],
    }

    # 原 6 题冒烟参照（映射至支持事件集合，命中其中任一支持事件即算冒烟命中）
    smoke6 = [
        (NO_KEYWORD_PROMPTS[0][0], NO_KEYWORD_PROMPTS[0][1], [e_wangcai_park, e_fuji_clean, e_santi, e_cof_new.event_id]),
        (NO_KEYWORD_PROMPTS[1][0], NO_KEYWORD_PROMPTS[1][1], [e_coke, e_dinner]),
        (NO_KEYWORD_PROMPTS[2][0], NO_KEYWORD_PROMPTS[2][1], [e_self, e_fe_dev, e_southbank, e_app_burnout]),
        (NO_KEYWORD_PROMPTS[3][0], NO_KEYWORD_PROMPTS[3][1], [e_thunder_desk, e_thunder_night, e_thunder_work, e_care_new.event_id]),
        (NO_KEYWORD_PROMPTS[4][0], NO_KEYWORD_PROMPTS[4][1], [e_trip_new.event_id, e_hokkaido_budget]),
        (NO_KEYWORD_PROMPTS[5][0], NO_KEYWORD_PROMPTS[5][1], [e_app_burnout, e_app_proto, e_app_code, e_app_advice]),
    ]
    return store, graph, eval_sets, smoke6


# ---------------------------------------------------------------------------
# 3. 评测与因果消融
# ---------------------------------------------------------------------------

def evaluate_method(
    name: str,
    recall_fn,
    store: EpisodicStore,
    eval_sets: Dict[str, List[RecallExample]],
    smoke6: List[Tuple[str, str, List[str]]],
) -> Dict[str, object]:
    metrics: Dict[str, object] = {"method": name}

    # 1) Direct recall
    d_exact = d_single = 0
    for ex in eval_sets["direct"]:
        res: RecallResult = recall_fn(ex.query)
        g = set(ex.gold_ids)
        p = set(res.event_ids)
        if g.issubset(p):
            d_exact += 1
        if g & p:
            d_single += 1
    n_d = len(eval_sets["direct"])
    metrics["direct_exact"] = round(d_exact / n_d, 4)
    metrics["direct_single"] = round(d_single / n_d, 4)

    # 2) Cross-event recall (重点：双事件全中率、单事件命中率、第二跳目标命中率)
    c_exact = c_single = c_target = c_jump = 0
    cross_details = []
    for ex in eval_sets["cross"]:
        res = recall_fn(ex.query)
        g = set(ex.gold_ids)
        p = set(res.event_ids)
        exact = g.issubset(p)
        single = bool(g & p)
        tgt = bool(ex.target_id and ex.target_id in p)
        c_exact += int(exact)
        c_single += int(single)
        c_target += int(tgt)
        c_jump += int(res.used_full_store_jump and tgt)
        cross_details.append({
            "query": ex.query,
            "gold": ex.gold_ids,
            "pred": res.event_ids,
            "exact": exact,
            "target_hit": tgt,
            "abstained": res.abstained,
            "p_valid": round(res.p_valid, 3),
            "via_jump": res.used_full_store_jump,
        })
    n_c = len(eval_sets["cross"])
    metrics["cross_exact"] = round(c_exact / n_c, 4)
    metrics["cross_single"] = round(c_single / n_c, 4)
    metrics["cross_target_hit"] = round(c_target / n_c, 4)
    metrics["cross_jump_hits"] = c_jump

    # 3) Hard negatives (误唤起率 FPR，越低越好)
    fp = 0
    for ex in eval_sets["hard_neg"]:
        res = recall_fn(ex.query)
        if not res.abstained and len(res.event_ids) > 0:
            fp += 1
    metrics["hard_neg_fpr"] = round(fp / len(eval_sets["hard_neg"]), 4)

    # 4) Superseded (最新事实命中率 & 旧值泄露率)
    s_hit = s_old_leak = 0
    for ex in eval_sets["superseded"]:
        res = recall_fn(ex.query)
        p = set(res.event_ids)
        if set(ex.gold_ids).issubset(p):
            s_hit += 1
        if any(eid in store and store.get(eid).status == "superseded" for eid in p):
            s_old_leak += 1
    metrics["superseded_latest_acc"] = round(s_hit / len(eval_sets["superseded"]), 4)
    metrics["superseded_old_leak"] = round(s_old_leak / len(eval_sets["superseded"]), 4)

    # 5) Insufficient evidence (正确拒答率)
    i_abs = 0
    for ex in eval_sets["insufficient"]:
        res = recall_fn(ex.query)
        if res.abstained or len(res.event_ids) == 0:
            i_abs += 1
    metrics["insufficient_abstain_acc"] = round(i_abs / len(eval_sets["insufficient"]), 4)

    # 6) Smoke-6 (原 6 题自然闲聊冒烟参照)
    sm_hit = 0
    for _tag, q, support_ids in smoke6:
        res = recall_fn(q)
        if set(res.event_ids) & set(support_ids):
            sm_hit += 1
    metrics["smoke6_hit"] = round(sm_hit / len(smoke6), 4)

    # 综合分（用于直观扫读，非替代单项指标）
    metrics["details_cross"] = cross_details
    return metrics


def run_causal_ablation(
    model: IterativeRecallNet,
    store: EpisodicStore,
    cross_examples: Sequence[RecallExample],
    mode: str = "hybrid",
    candidate_budget: int = 16,
    abstain_thresh: float = 0.5,
) -> Dict[str, float]:
    """必做因果消融：移掉桥接事件（drop_bridge）vs 移掉无关事件（drop_unrelated）。"""
    all_active = store.active_events()
    full_tgt = drop_bridge_tgt = drop_unrel_tgt = 0
    n = len(cross_examples)

    full_graph = SparseAssociativeGraph(store, max_neighbors=16)

    for ex in cross_examples:
        assert ex.bridge_id and ex.target_id
        # 1) 完整事件库
        r_full = model.recall(
            store, full_graph, ex.query, mode=mode, candidate_budget=candidate_budget, abstain_thresh=abstain_thresh
        )
        if ex.target_id in r_full.event_ids:
            full_tgt += 1

        # 2) 移掉桥接事件
        sub_b = store.clone_without([ex.bridge_id])
        g_b = SparseAssociativeGraph(sub_b, max_neighbors=16)
        r_b = model.recall(
            sub_b, g_b, ex.query, mode=mode, candidate_budget=candidate_budget, abstain_thresh=abstain_thresh
        )
        if ex.target_id in r_b.event_ids:
            drop_bridge_tgt += 1

        # 3) 移掉一个无关事件（既非桥接也非目标，且与查询相似度最低的事件之一）
        q0 = store.encoder.encode_one(ex.query)
        unrel = min(
            (e for e in all_active if e.event_id not in (ex.bridge_id, ex.target_id)),
            key=lambda e: float(torch.dot(q0, e.embedding).item()),
        )
        sub_u = store.clone_without([unrel.event_id])
        g_u = SparseAssociativeGraph(sub_u, max_neighbors=16)
        r_u = model.recall(
            sub_u, g_u, ex.query, mode=mode, candidate_budget=candidate_budget, abstain_thresh=abstain_thresh
        )
        if ex.target_id in r_u.event_ids:
            drop_unrel_tgt += 1

    return {
        "full_target_hit": round(full_tgt / n, 4),
        "drop_bridge_target_hit": round(drop_bridge_tgt / n, 4),
        "drop_unrelated_target_hit": round(drop_unrel_tgt / n, 4),
        "bridge_causal_drop": round((full_tgt - drop_bridge_tgt) / n, 4),
        "unrelated_drop": round((full_tgt - drop_unrel_tgt) / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== E1 可学习循环回想实测（dev={dev}, budget={args.budget}, seeds={seeds}）===", flush=True)

    encoder = FrozenTextEncoder("BAAI/bge-small-zh-v1.5", device=dev)
    tr_store, tr_graph, tr_examples = build_disjoint_train_world(encoder)
    ev_store, ev_graph, eval_sets, smoke6 = build_ayuan_eval_world(encoder)
    print(
        f"训练集：{len(tr_store.all_events())} 事件（{len(tr_examples)} 样本，12 个独立主题）\n"
        f"评测集：{len(ev_store.all_events())} 事件（24 篇阿远日记切分 + 变更更新，active={len(ev_store.active_events())}）",
        flush=True,
    )

    # 校准 3 个非学习公平基线（仅用训练集，绝不碰评测集）
    b_oneshot = OneShotBaseline()
    b_oneshot.calibrate(tr_store, tr_examples)

    b_fixed2 = Fixed2HopGraphBaseline()
    b_fixed2.calibrate(tr_store, tr_graph, tr_examples)

    b_plain2 = PlainTwoRoundBaseline()
    b_plain2.calibrate(tr_store, tr_examples)
    print(
        f"基线阈值校准（仅训练集）: one_shot={b_oneshot.thresh1:.2f}/{b_oneshot.thresh2:.2f} | "
        f"fixed_2hop={b_fixed2.thresh1:.2f}/{b_fixed2.thresh2:.2f} | "
        f"plain_2round={b_plain2.thresh1:.2f}/{b_plain2.thresh2:.2f} (alpha={b_plain2.alpha})",
        flush=True,
    )

    # 先评 3 个确定性基线
    base_results = {}
    for name, fn in [
        ("one_shot", lambda q: b_oneshot.recall(ev_store, q, top_k=2)),
        ("fixed_2hop_graph", lambda q: b_fixed2.recall(ev_store, ev_graph, q)),
        ("plain_2round", lambda q: b_plain2.recall(ev_store, q)),
    ]:
        base_results[name] = evaluate_method(name, fn, ev_store, eval_sets, smoke6)

    # 多种子训练与评测学习版（hybrid / graph_only / full_only）+ 因果消融 + 跳图巩固验证
    seed_runs = []
    for sd in seeds:
        torch.manual_seed(sd)
        net = IterativeRecallNet(emb_dim=encoder.dim, hidden_dim=32, low_rank=16).to(dev)
        losses = train_iterative_recall(
            net, tr_store, tr_graph, tr_examples,
            epochs=args.epochs, lr=args.lr, candidate_budget=args.budget, seed=sd,
        )
        abs_t = net.calibrate_abstain_thresh(tr_store, tr_graph, tr_examples, mode="hybrid", candidate_budget=args.budget)
        print(f"[seed={sd}] losses={losses} | calibrated abstain_thresh={abs_t:.2f}", flush=True)

        res_sd = {}
        for name, mode in [
            ("learned_hybrid", "hybrid"),
            ("learned_graph_only", "graph_only"),
            ("learned_full_only", "full_only"),
        ]:
            res_sd[name] = evaluate_method(
                name,
                lambda q, m=mode, t=abs_t: net.recall(ev_store, ev_graph, q, mode=m, candidate_budget=args.budget, abstain_thresh=t),
                ev_store, eval_sets, smoke6,
            )
        causal_sd = run_causal_ablation(
            net, ev_store, eval_sets["cross"], mode="hybrid", candidate_budget=args.budget, abstain_thresh=abs_t
        )

        # Priority #1 专项验证：移除所有指向 u_care_new 的图边
        jump_probe_q = eval_sets["cross"][0].query
        target_id = eval_sets["cross"][0].target_id
        g_unlinked = SparseAssociativeGraph(ev_store, max_neighbors=16)
        for u_id in list(g_unlinked.adj.keys()):
            g_unlinked.adj[u_id].pop(target_id, None)
        g_unlinked.adj[target_id] = {}
        r_g0 = net.recall(ev_store, g_unlinked, jump_probe_q, mode="graph_only", candidate_budget=args.budget, abstain_thresh=abs_t)
        r_hj = net.recall(ev_store, g_unlinked, jump_probe_q, mode="hybrid", candidate_budget=args.budget, abstain_thresh=abs_t)
        used_bridge = r_hj.event_ids[0] if len(r_hj.event_ids) >= 2 else eval_sets["cross"][0].bridge_id
        if target_id in r_hj.event_ids:
            g_unlinked.reinforce_path(r_hj.event_ids, delta=0.8)
        r_g1 = net.recall(ev_store, g_unlinked, jump_probe_q, mode="graph_only", candidate_budget=args.budget, abstain_thresh=abs_t)
        jp = {
            "bridge_id": used_bridge,
            "target_id": target_id,
            "graph_only_before_consol": r_g0.event_ids,
            "hybrid_jump_recall": r_hj.event_ids,
            "hybrid_used_jump": r_hj.used_full_store_jump,
            "graph_only_after_consol": r_g1.event_ids,
            "consolidated_edge_created": target_id in g_unlinked.adj.get(used_bridge, {}),
        }
        seed_runs.append({
            "seed": sd,
            "abstain_thresh": abs_t,
            "losses": losses,
            "results": {k: {mk: mv for mk, mv in v.items() if mk != "details_cross"} for k, v in res_sd.items()},
            "causal_ablation": causal_sd,
            "jump_probe": jp,
            "cross_details_hybrid": res_sd["learned_hybrid"]["details_cross"],
        })

    # 聚合多种子均值
    metric_keys = [
        "cross_exact", "cross_single", "cross_target_hit", "direct_exact",
        "hard_neg_fpr", "superseded_latest_acc", "superseded_old_leak",
        "insufficient_abstain_acc", "smoke6_hit",
    ]
    summary = {k: {mk: v[mk] for mk in metric_keys} for k, v in base_results.items()}
    for name in ("learned_hybrid", "learned_graph_only", "learned_full_only"):
        summary[name] = {
            mk: round(sum(r["results"][name][mk] for r in seed_runs) / len(seed_runs), 4)
            for mk in metric_keys
        }
    causal_mean = {
        k: round(sum(r["causal_ablation"][k] for r in seed_runs) / len(seed_runs), 4)
        for k in seed_runs[0]["causal_ablation"]
    }

    print(f"\n=== E1 核心指标对比（{len(seeds)} 种子均值，零主题泄露，纯事件 ID 评测）===")
    header = f"{'方法':<20} | {'跨事件全中':>8} | {'跨事件单中':>8} | {'二跳目标':>8} | {'直接回想':>8} | {'难负例FPR↓':>9} | {'已变更↑':>7} | {'缺证拒答↑':>8} | {'冒烟6题':>7}"
    print(header)
    print("-" * len(header))
    for name in ("one_shot", "fixed_2hop_graph", "plain_2round", "learned_hybrid", "learned_graph_only", "learned_full_only"):
        m = summary[name]
        print(
            f"{name:<20} | {m['cross_exact']:>8.1%} | {m['cross_single']:>8.1%} | "
            f"{m['cross_target_hit']:>8.1%} | {m['direct_exact']:>8.1%} | "
            f"{m['hard_neg_fpr']:>9.1%} | {m['superseded_latest_acc']:>7.1%} | "
            f"{m['insufficient_abstain_acc']:>8.1%} | {m['smoke6_hit']:>7.1%}",
            flush=True,
        )

    print(f"\n=== 必做因果消融（learned_hybrid，{len(seeds)} 种子均值）===")
    print(
        f"完整库: {causal_mean['full_target_hit']:.1%} | "
        f"移掉桥接事件: {causal_mean['drop_bridge_target_hit']:.1%} (Δ={-causal_mean['bridge_causal_drop']:+.1%}) | "
        f"移掉无关事件: {causal_mean['drop_unrelated_target_hit']:.1%} (Δ={-causal_mean['unrelated_drop']:+.1%})",
        flush=True,
    )
    jump_ok = sum(1 for r in seed_runs if r["jump_probe"]["consolidated_edge_created"] and r["jump_probe"]["target_id"] in r["jump_probe"]["graph_only_after_consol"])
    print(f"\n=== Priority #1 未连边跳图→巩固成边通过率：{jump_ok}/{len(seeds)} ===", flush=True)

    elapsed = round(time.time() - t0, 1)
    out_path = ROOT / f"runs/{time.strftime('%Y%m%d-%H%M%S')}_assoc_e1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E1_associative_memory",
        "encoder": "BAAI/bge-small-zh-v1.5",
        "budget": args.budget,
        "epochs": args.epochs,
        "seeds": seeds,
        "elapsed_s": elapsed,
        "train_events": len(tr_store.all_events()),
        "train_examples": len(tr_examples),
        "eval_events": len(ev_store.all_events()),
        "eval_active_events": len(ev_store.active_events()),
        "summary_mean": summary,
        "causal_ablation_mean": causal_mean,
        "jump_consolidate_pass": f"{jump_ok}/{len(seeds)}",
        "seed_runs": seed_runs,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out_path}（耗时 {elapsed}s）", flush=True)


if __name__ == "__main__":
    main()
