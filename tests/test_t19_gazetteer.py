"""T19: place-resolution fixes for the migrant variables (at least two cases per error class, incl. c12)."""
import numpy as np
import pytest

from src.extract_rules import _migrant, origin_fields
from src.gazetteer import (court_prefecture, origin_text_clean, prefecture_on_date, province_of_prefecture,
                           resolve_place)


def pref(raw, prov=None, city=None):
    return resolve_place(raw, prov, city)[1]


# 1a. municipality followed by an unresolvable part -> the municipality (not the pre-1997 Sichuan 重庆市)
@pytest.mark.parametrize("raw", ["重庆市某县", "重庆市原万盛区", "重庆市", "重庆市北部新区"])
def test_municipality(raw):
    p, f, _, _ = resolve_place(raw, "重庆市", "重庆市")
    assert (p, f) == ("重庆市", "重庆市")


# 1b. districts created after 2016 belong to their prefecture
@pytest.mark.parametrize("raw,city", [("浙江省杭州市临安区", "杭州市"), ("青岛市即墨区", "青岛市"), ("宜宾市叙州区", "宜宾市"),
                                      ("福州市长乐区", "福州市"), ("陕西省汉中市南郑区", "汉中市"), ("九江市柴桑区", "九江市")])
def test_new_district(raw, city):
    assert pref(raw) == city


# 2. place names inside agency names are not origins
@pytest.mark.parametrize("raw", ["长春市公安局双阳区分局", "吉林省长春市公安局双阳区分局石溪派出所", "长春市公安局宽城区分局孟家桥派出所管区"])
def test_agency_names(raw):
    assert pref(raw, "吉林省", "长春市") == "长春市"


# 3. an explicit prefecture wins over a later fragment that belongs elsewhere
@pytest.mark.parametrize("raw,city", [("中山市东区", "中山市"), ("广东省湛江市东海岛", "湛江市"),
                                      ("湖北省武汉东湖新技术开发区", "武汉市"), ("福建省福州市古田支路", "福州市"),
                                      ("山东省滨州市北海新区", "滨州市"), ("长春市乐山镇", "长春市")])
def test_explicit_prefecture_first(raw, city):
    assert pref(raw) == city


def test_county_still_refines_within_prefecture():
    p, f, c, _ = resolve_place("湖南省衡阳市耒阳市")
    assert (f, c) == ("衡阳市", "耒阳市")
    assert pref("古田县") == "宁德市"          # a county alone still resolves


# 4. a missing (NaN) court province is missing, not "different"; it is taken from the court prefecture
def test_nan_court_province():
    assert _migrant("重庆市", "重庆市", np.nan, "重庆市", "city") is None
    assert province_of_prefecture("重庆市") == "重庆市" and province_of_prefecture("莱芜市") == "山东省"
    out = origin_fields({"birthplace_raw": "重庆市开县"}, np.nan, "重庆市", "2015-03-01")
    assert out["migrant_city_nores"] is False
    out = origin_fields({"native_raw": "莱芜市莱城区人"}, float("nan"), "莱芜市", "2014-06-01")
    assert out["migrant_city_nores"] is False


# 5. Laiwu belongs to Jinan from 2019-01-09 (c12: "出生于济南市钢城区", Jinan Gangcheng court, 2019)
def test_c12_jinan_gangcheng_2019():
    assert court_prefecture(None, None, "山东省", "济南市钢城区人民法院") == "济南市"
    out = origin_fields({"birthplace_raw": "济南市钢城区"}, "山东省", "济南市", "2019-05-20")
    assert out["migrant_city_nores"] is False


@pytest.mark.parametrize("raw,court,date,migrant", [
    ("济南市莱芜区人", "济南市", "2019-03-01", False),     # new district, Jinan court
    ("莱芜市莱城区人", "济南市", "2019-03-01", False),     # old name, after the merger
    ("莱芜市莱城区人", "济南市", "2018-06-01", True),      # before the merger Laiwu is another prefecture
    ("莱芜市钢城区人", "莱芜市", "2015-06-01", False),
])
def test_laiwu_jinan_by_date(raw, court, date, migrant):
    assert origin_fields({"native_raw": raw}, "山东省", court, date)["migrant_city_nores"] is migrant
    assert prefecture_on_date("莱芜市", "2019-01-09") == "济南市" and prefecture_on_date("莱芜市", "2019-01-08") == "莱芜市"


# 6. address words in an origin field: not an origin
@pytest.mark.parametrize("raw,clean", [("现居住地：白山市", None), ("家庭住址为河北省秦皇岛市昌黎县", None),
                                       ("被捕前居住地辽宁省大连市", None), ("河南省卫辉市暂住太原市迎泽区", "河南省卫辉市"),
                                       ("四川省高县居住地重庆市南岸区", "四川省高县"), ("湖南省衡阳市", "湖南省衡阳市")])
def test_origin_text_clean(raw, clean):
    assert origin_text_clean(raw) == clean


def test_address_in_hukou_falls_back_to_next_source():
    out = origin_fields({"hukou_raw": "现居住地：白山市", "native_raw": "吉林省通化市人"}, "吉林省", "白山市", "2015-01-01")
    assert out["origin_source"] == "native" and out["migrant_city_nores"] is True
    out = origin_fields({"hukou_raw": "现居住地：白山市"}, "吉林省", "白山市", "2015-01-01")
    assert out["migrant_city_nores"] is None


# 3 (limits). a full county-level name still wins over a named prefecture: it is more specific
@pytest.mark.parametrize("raw,city", [("吉林榆树市", "长春市"), ("河北肃宁县", "沧州市"), ("安徽省巢湖市和县", "马鞍山市"),
                                      ("吉林大安市", "白城市")])
def test_full_county_name_still_wins(raw, city):
    assert pref(raw) == city


# 7. (found in T19) prefecture conversions areacodes does not link; a NaN origin prefecture is not "different"
@pytest.mark.parametrize("raw,city", [("青海省海东地区", "海东市"), ("西藏昌都地区昌都县人", "昌都市"),
                                      ("西藏山南地区贡嘎县人", "山南市"), ("新疆吐鲁番地区", "吐鲁番市")])
def test_unlinked_prefecture_conversions(raw, city):
    assert pref(raw) == city


def test_nan_origin_prefecture_is_missing():
    assert _migrant("青海省", float("nan"), "青海省", "西宁市", "city") is None
