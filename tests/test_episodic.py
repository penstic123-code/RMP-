"""情景记忆：跨会话按线索找回陈述句，不存问句，不重现本会话。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from episodic import Episodic


def test_recall_across_sessions():
    E = Episodic()
    for l in ['My name is Tovaku.', 'I have a cat named Mirafe.', 'What is my name?']:
        E.write(l)
    assert E.recall('What is my name?') == []                 # 同一会话不重现
    E.new_session()
    assert E.recall('What is my cat\'s name?')[-1] == 'I have a cat named Mirafe.' or 'I have a cat named Mirafe.' in E.recall("What is my cat's name?")
    assert all(not r.endswith('?') for r in E.recall('What is my name?'))
    assert E.recall('Hi again!') == []
