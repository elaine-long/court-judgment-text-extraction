"""T09d: jiejie counts only a signed plea pledge, not "具结悔过" or cross-word hits."""
import pytest

from src.extract_rules import _JIEJIE, extract_doc

# false positives found while checking the rule, plus other cross-word hits
FALSE = [
    "被告人××网络工具结识了被害人××，后双方确定了男女朋友关系",
    "不需要判处刑罚的，可以免予刑事处罚，但是可以根据案件的不同情况，予以训诫或者责令具结悔过、赔礼道歉",
    "人民法院审理民事案件，除适用上述规定外，还可以予以训诫、责令具结悔过、收缴进行非法活动的财务",
    "还可以予以训诫、责令具结悔过、收缴进行非法活动的财物和非法所得，并可以依照法律规定处以罚款",
    "价格认证中心出具结论书，被盗电动车价值1048元",
    "两被告人持管制刀具结伙抢劫，可酌情从重处罚",
    "燃气表具结构说明，抄表记录，扣押决定书",
    "2004年7月4日，因盗窃被责令具结悔过；2004年7月16日，因盗窃被行政拘留七日",
]
TRUE = [
    "被告人柴某对公诉机关指控的事实、罪名及量刑建议没有异议且签字具结，在开庭审理过程中亦无异议。",
    "公诉机关提交了被告人石某的供述，被害人席某的陈述，认罪认罚具结书及公安机关制作的到案经过",
    "并签署了认罪认罚具结悔过书。上述事实，公诉机关提供证据材料如下。",
    "被告人郑某对指控事实、罪名及量刑建议无异议且��字具结，在开庭审理过程中亦无异议。",   # 签 lost to encoding
    "被告人付某对公诉机关指控的犯罪事实及罪名没有异议，且认罪具结，在开庭审理过程中亦无异议。",
]


@pytest.mark.parametrize("text", FALSE)
def test_not_a_plea_pledge(text):
    assert not _JIEJIE.search(text)


@pytest.mark.parametrize("text", TRUE)
def test_plea_pledge(text):
    assert _JIEJIE.search(text)


def test_doc_level_flags():
    base = ("某法院刑事判决书公诉机关某检察院。被告人甲，男，住某县。因涉嫌抢劫被逮捕。现羁押于某看守所。"
            "某检察院以某号起诉书指控被告人甲犯抢劫罪，于某日提起公诉。公诉机关指控：甲抢得现金1000元。"
            "本院认为，被告人甲构成抢劫罪。判决如下：被告人甲犯抢劫罪，判处有期徒刑三年。")
    d, _ = extract_doc(1, base + "附：刑法第三十七条……予以训诫或者责令具结悔过、赔礼道歉、赔偿损失。")
    assert not d["jiejie"] and not d["leniency_proc"]
    d, _ = extract_doc(1, base.replace("本院认为", "被告人对量刑建议没有异议且签字具结。本院认为"))
    assert d["jiejie"] and d["leniency_proc"] and not d["plea_formal"]
