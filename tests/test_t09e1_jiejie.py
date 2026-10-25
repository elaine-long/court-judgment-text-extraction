"""T09e1: "认罪认罚 … 具结悔过书" within 30 characters is a plea pledge; "责令具结悔过" is not."""
import pytest

from src.extract_rules import has_jiejie

TRUE = [
    "案件审理中，被告人李某表示自愿认罪认罚，并提交了具结悔过书。本院认为",            # T09d sample, 6 of 20
    "被告人张某、罗某表示自愿认罪认罚，并提交了具结悔过书。",                         # two defendants
    "被告人王某提交了具结悔过书，表示自愿认罪认罚。",                                 # plea after the pledge
    "被告人赵某自愿认罪认罚，愿意接受处罚，当庭向被害人道歉并递交了具结悔过书。",       # gap < 30
    "被告人对指控事实无异议且签字具结，在开庭审理过程中亦无异议。",                    # T09d rule still applies
]
FALSE = [
    "不需要判处刑罚的，可以免予刑事处罚，但是可以根据案件的不同情况，予以训诫或者责令具结悔过、赔礼道歉",
    "被告人自愿认罪认罚。附：刑法第三十七条……予以训诫或者责令具结悔过、赔礼道歉、赔偿损失",   # 责令 excluded
    "被告人孙某提交了具结悔过书。" + "其他情节" * 10 + "被告人孙某自愿认罪认罚。",            # gap > 30
    "2004年7月4日，因盗窃被责令具结悔过；2004年7月16日，因盗窃被行政拘留七日",
]


@pytest.mark.parametrize("text", TRUE)
def test_plea_pledge(text):
    assert has_jiejie(text)


@pytest.mark.parametrize("text", FALSE)
def test_not_a_plea_pledge(text):
    assert not has_jiejie(text)


def test_gap_boundary():
    pad = "某" * 30
    assert has_jiejie("认罪认罚" + pad + "具结悔过书")          # gap exactly 30
    assert not has_jiejie("认罪认罚" + pad + "某具结悔过书")    # gap 31
