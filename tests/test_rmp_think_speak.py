"""思说解耦 RMP（ThinkSpeakRMP）核心单元测试。"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rmp_think_speak import (
    RMPEpisodicBank,
    ThinkSpeakRMP,
    decode_bytes,
    encode_bytes,
)


def test_byte_codec_roundtrip():
    txt = "答复：先找小李，再找陈阿姨｜依据：最新交接记录"
    ids = encode_bytes(txt)
    assert decode_bytes(ids) == txt


def test_think_speak_rmp_forward_and_lockstep_vs_decoupled():
    torch.manual_seed(0)
    model = ThinkSpeakRMP(D=64, N=16, k=2, G=4, pd=32, pm=8, phidden=64, qr=16, mem_dim=32)
    bank = RMPEpisodicBank()
    bank.add("m1", "最新记录：大门钥匙现在交给小李保管。", day=1)
    bank.add("m2", "最新记录：暴雨打雷时要联系陈阿姨照看宠物。", day=2)

    # 1. Lockstep 模式（max_think_total=0）绝不执行内部思考步
    out_lock = model.generate("我现在该找谁拿大门钥匙？", bank=bank, max_new_tokens=16, max_think_total=0)
    assert out_lock["think_steps_total"] == 0
    assert out_lock["recalled_ids"] == []

    # 2. 强制触发思考（emit_thresh=1.1）验证开口前思考 + 记忆读入 + 正常吐字不崩溃
    out_think = model.generate(
        "我现在该找谁拿大门钥匙？",
        bank=bank,
        max_new_tokens=16,
        max_think_total=3,
        max_think_consec=2,
        emit_thresh=1.1,
    )
    assert out_think["think_steps_total"] == 3
    assert out_think["pre_speech_thinks"] == 2  # 开口前连想 2 步后强制吐字，句中还能再停 1 步
    assert out_think["mid_speech_thinks"] <= 1


def test_episodic_bank_supersede_and_delete():
    bank = RMPEpisodicBank()
    bank.add("k1", "大门钥匙交给陈阿姨。", day=1)
    assert [e.event_id for e in bank.active_entries()] == ["k1"]
    bank.supersede("k1", "k2", "大门钥匙转交给小李。", day=2)
    assert [e.event_id for e in bank.active_entries()] == ["k2"]
    bank.delete("k2")
    assert bank.active_entries() == []
