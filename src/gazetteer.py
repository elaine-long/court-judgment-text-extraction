"""Gazetteer of county-level-and-above divisions (1981-2025, with history) and place resolution.

Source: yescallop/areacodes (CC0), result.csv, pinned commit in data/external/areacodes_VERSION.txt.
Fetch (data/ is not in git):
    curl -o data/external/areacodes_result.csv \
      https://raw.githubusercontent.com/yescallop/areacodes/e2c2b2e1fb4ce1fb76e879aa0b80e229adaf950c/result.csv
Every record is normalised to its 2016 successor (`code_2016`), so historical names (蓟县, 襄樊市,
巢湖地区的庐江县) map to the prefecture/province they belonged to during the study window.
"""
import re
from functools import lru_cache

import pandas as pd

from src import paths

RAW = paths.DATA / "external" / "areacodes_result.csv"
GAZ = paths.DATA / "gazetteer.parquet"
REF_YEAR = 2016
WINDOW = (2012, 2020)
MUNI_PREFIX = {"11": "北京市", "12": "天津市", "31": "上海市", "50": "重庆市"}
PROV_SHORT = {"内蒙古自治区": ["内蒙古", "内蒙"], "广西壮族自治区": ["广西"], "西藏自治区": ["西藏"],
              "宁夏回族自治区": ["宁夏"], "新疆维吾尔自治区": ["新疆"], "香港特别行政区": ["香港"],
              "澳门特别行政区": ["澳门"], "黑龙江省": ["黑龙江"]}
LOCAL_WORDS = ("本市", "本区", "本县", "本省", "本镇", "本乡", "本地", "本辖区")
_ETHNIC_AUTONOMY = re.compile(r"(?:[一-龥]{1,6}?族)+自治(?:县|州|旗)$|各族自治县$")
_PREFIX_JUNK = re.compile(r"^(?:中华人民共和国|中国)")
# T19: place names inside agency names are not places of origin ("长春市公安局双阳区分局" -> 公安县 was a bug)
_AGENCY = re.compile(r"公安局|公安分局|分局|派出所|看守所|拘留所|检察院|人民法院|法院|监狱|戒毒所")
# T19: an origin field that is really an address (codebook: an address is never the place of origin)
_ADDRESS_START = re.compile(r"^[：:，,、\s]*(?:现居住地|居住地|家庭住址|家庭地址|被捕前居住地|捕前居住地|现住址|住址|住所地|暂住地|现住|暂住|家住)")
_ADDRESS_MID = re.compile(r"(?:现居住地|居住地|家庭住址|现住址|住址|住所地|暂住地|现住|暂住|家住|捕前住)")
# T19: district names that exist in many cities; never let them override a prefecture named before them
GENERIC_DISTRICTS = {"东区", "西区", "南区", "北区", "中区", "城区", "郊区", "矿区", "新区", "市中区", "城中区", "城东区",
                     "城西区", "城南区", "城北区", "市辖区"}
# T19: Laiwu (莱芜市) was merged into Jinan (济南市) by the State Council approval of 9 January 2019
MERGERS = {"莱芜市": ("济南市", "2019-01-09")}


# ----------------------------------------------------------------------------- build
def _short_names(name, level):
    out = []
    if level == "省级":
        out += PROV_SHORT.get(name, [])
        if name.endswith(("省", "市")) and len(name) >= 3:
            out.append(name[:-1])
        return out
    base = _ETHNIC_AUTONOMY.sub("", name)
    if base != name and len(base) >= 2:
        out.append(base)
    for suf in ("地区", "林区", "特区", "市", "县", "区", "旗", "盟"):
        if base.endswith(suf) and len(base) - len(suf) >= 2:
            out.append(base[: -len(suf)])
            break
    return [x for x in dict.fromkeys(out) if x != name]


def build_gazetteer():
    g = pd.read_csv(RAW, dtype=str)
    g.columns = [c.strip("﻿") for c in g.columns]
    g = g.rename(columns={"代码": "code", "一级行政区": "province_at", "二级行政区": "parent_at",
                          "名称": "full_name", "级别": "level", "状态": "status", "启用时间": "valid_from",
                          "变更/弃用时间": "valid_to", "新代码": "new_codes"})
    g["valid_from"] = g["valid_from"].astype(int)
    g["valid_to"] = pd.to_numeric(g["valid_to"]).astype("Int64")

    def valid_in(r, y):
        return r.valid_from <= y and (pd.isna(r.valid_to) or y < r.valid_to)

    by_code = {}
    for r in g.itertuples():
        by_code.setdefault(r.code, []).append(r)

    def rec_at(code, y):
        for r in by_code.get(code, []):
            if valid_in(r, y):
                return r
        return None

    @lru_cache(maxsize=None)
    def successor(code, depth=0):
        """2016 code for an old code, following 新代码 links."""
        if rec_at(code, REF_YEAR) is not None:
            return code
        if depth > 10:
            return None
        for r in sorted(by_code.get(code, []), key=lambda x: -x.valid_from):
            if isinstance(r.new_codes, str) and r.new_codes:
                first = re.sub(r"\[\d{4}\]", "", r.new_codes.split(";")[0])
                s = successor(first, depth + 1)
                if s:
                    return s
        return None

    def prefecture_of(code16):
        r = rec_at(code16, REF_YEAR)
        if r is None:
            return None, None
        if r.level == "省级":
            return None, None
        if code16[:2] in MUNI_PREFIX:
            return MUNI_PREFIX[code16[:2]], code16[:2] + "0000"
        if r.level == "地级":
            return r.full_name, code16
        p = rec_at(code16[:4] + "00", REF_YEAR)
        if p is not None and p.level == "地级":
            return p.full_name, p.code
        return r.full_name, code16  # 省直辖县级行政单位 (仙桃市, 济源市 …)

    # T19: 2016 prefectures by (province, name stem), for conversions areacodes does not link
    # (海东地区 -> 海东市 2013, 昌都/日喀则/林芝/山南地区 -> 市, 吐鲁番/哈密地区 -> 市, 那曲市 2017 -> 那曲地区)
    def stem(n):
        return re.sub(r"(?:地区|自治州|盟|市)$", "", n)
    pref16 = {}
    for r in g.itertuples():
        if r.level == "地级" and valid_in(r, REF_YEAR):
            pref16[(r.code[:2], stem(r.full_name))] = (r.full_name, r.code)

    def pref_by_stem(code, name):
        return pref16.get((code[:2], stem(name)), (None, None))

    rows = []
    for r in g.itertuples():
        c16 = successor(r.code) if r.level != "省级" else r.code
        prov_rec = rec_at((c16 or r.code)[:2] + "0000", REF_YEAR)
        province = prov_rec.full_name if prov_rec is not None else r.province_at
        pref, pref_code = prefecture_of(c16) if c16 else (None, None)
        if c16 is None and r.level == "县级" and r.valid_from > REF_YEAR:
            # T19: a district created after 2016 (即墨区 2017, 叙州区 2018, 济南市莱芜区 2019 …) has no 2016 code;
            # it belongs to the prefecture of its own code's first four digits (as that prefecture stood in 2016)
            parent = rec_at(r.code[:4] + "00", REF_YEAR)
            if r.code[:2] in MUNI_PREFIX:
                pref, pref_code = MUNI_PREFIX[r.code[:2]], r.code[:2] + "0000"
            elif parent is not None and parent.level == "地级":
                pref, pref_code = parent.full_name, parent.code
        if pref is None and r.level == "地级":
            pref, pref_code = pref_by_stem(r.code, r.full_name)
        if pref is None and r.level == "县级" and r.code[:2] not in MUNI_PREFIX:
            parents = by_code.get(r.code[:4] + "00", [])
            if parents:
                pref, pref_code = pref_by_stem(r.code, parents[0].full_name)
        in_window = r.valid_from <= WINDOW[1] and (pd.isna(r.valid_to) or r.valid_to > WINDOW[0])
        vt = "" if pd.isna(r.valid_to) else str(int(r.valid_to) - 1)
        base = {"full_name": r.full_name, "level": {"省级": "province", "地级": "prefecture", "县级": "county"}[r.level],
                "code": r.code, "code_2016": c16,
                "county_code": c16 if r.level == "县级" else None,
                "prefecture": pref, "prefecture_code": pref_code,
                "province": province, "province_code": (c16 or r.code)[:2] + "0000",
                "valid_from": r.valid_from, "valid_to": r.valid_to, "valid_years": f"{r.valid_from}-{vt}",
                "in_window": bool(in_window)}
        rows.append({"name": r.full_name, "is_short": False, **base})
        for s in _short_names(r.full_name, r.level):
            rows.append({"name": s, "is_short": True, **base})
    out = pd.DataFrame(rows)
    out.to_parquet(GAZ, index=False)
    return out


# ----------------------------------------------------------------------------- resolve
_INDEX = None


def _load():
    global _INDEX
    if _INDEX is None:
        gz = pd.read_parquet(GAZ) if GAZ.exists() else build_gazetteer()
        idx = {}
        for r in gz.to_dict("records"):
            idx.setdefault(r["name"], []).append(r)
        _INDEX = (idx, max(len(k) for k in idx))
    return _INDEX


def _match_at(s, pos):
    idx, maxlen = _load()
    for L in range(min(maxlen, len(s) - pos), 1, -1):
        cands = idx.get(s[pos:pos + L])
        if cands:
            return L, cands
    return 0, None


def _prefer_window(cands):
    w = [c for c in cands if c["in_window"]]
    return w or cands


def resolve_place(raw, court_province=None, court_prefecture=None):
    """Return (province, prefecture, county, status).

    status: resolved | province_only | ambiguous_resolved_by_court_prov | ambiguous | unresolved | local_word
    """
    if not isinstance(raw, str):
        return None, None, None, "unresolved"
    s = _AGENCY.sub("", _PREFIX_JUNK.sub("", raw.strip("：: ，,。")))
    s = re.sub(r"人$", "", s)
    if s.startswith(LOCAL_WORDS):
        if s.startswith("本省"):
            return court_province, None, None, "local_word"
        return court_province, court_prefecture, None, "local_word"
    tokens, pos = [], 0
    while pos < len(s) and len(tokens) < 3:
        L, cands = _match_at(s, pos)
        if not L:
            break
        tokens.append(cands)
        pos += L
        if any(c["level"] == "county" for c in cands) and all(c["level"] == "county" for c in cands):
            break
    if not tokens:
        return None, None, None, "unresolved"
    prov_tok = [c for t in tokens for c in t if c["level"] == "province"]
    provs = {c["province"] for c in prov_tok}
    # the most specific token decides; earlier tokens constrain it
    cands = None
    for t in tokens:
        muni = [c for c in t if c["level"] == "province" and c["full_name"] in MUNI_PREFIX.values()]
        if muni:
            # T19: a municipality is also its own prefecture; its pre-1997 Sichuan record (重庆市) is ignored
            sub = [_muni_as_prefecture(muni[0])]
        else:
            sub = [c for c in t if c["level"] != "province"]
        if not sub:
            continue
        if provs:
            sub = [c for c in sub if c["province"] in provs] or sub
        if cands:
            prefs = {c["prefecture_code"] for c in cands}
            narrowed = [c for c in sub if c["prefecture_code"] in prefs]
            if not narrowed and all(c["prefecture"] for c in cands) and all(
                    c["is_short"] or c["full_name"] in GENERIC_DISTRICTS for c in sub):
                # T19: an explicit prefecture already named in the text wins over a later fragment that belongs
                # elsewhere only through a short name or a generic district name ("湛江市东海岛" is not 东海县,
                # "中山市东区" is not 攀枝花市东区). A full county-level name still wins ("吉林榆树市" -> 长春市,
                # "巢湖市和县" -> 马鞍山市): it is more specific, and the named prefecture may be outdated.
                break
            sub = narrowed or sub
        cands = _prefer_window(sub)
    if not cands:
        if len(provs) == 1:
            return provs.pop(), None, None, "province_only"
        return None, None, None, "unresolved"
    keys = {(c["province"], c["prefecture"]) for c in cands if c["prefecture"]}
    county = None
    cty = {c["full_name"] for c in cands if c["level"] == "county"}
    if len(cty) == 1:
        county = next(iter(cty))
    if len(keys) == 1:
        p, f = next(iter(keys))
        return p, f, county, "resolved"
    if court_province:
        k2 = {k for k in keys if k[0] == court_province}
        if len(k2) == 1:
            p, f = next(iter(k2))
            return p, f, county, "ambiguous_resolved_by_court_prov"
    ps = {k[0] for k in keys}
    return (next(iter(ps)) if len(ps) == 1 else None), None, None, "ambiguous"


def _muni_as_prefecture(c):
    code = {v: k for k, v in MUNI_PREFIX.items()}[c["full_name"]] + "0000"
    return {**c, "level": "prefecture", "prefecture": c["full_name"], "prefecture_code": code, "in_window": True}


def origin_text_clean(raw):
    """Origin text without address parts (T19, codebook §1: an address is never the place of origin).

    Returns None when the text starts as an address ("现居住地：白山市"); cuts a trailing address
    ("河南省卫辉市暂住太原市迎泽区" -> "河南省卫辉市")."""
    if not isinstance(raw, str):
        return raw
    if _ADDRESS_START.match(raw):
        return None
    m = _ADDRESS_MID.search(raw)
    return raw[:m.start()] if m and m.start() >= 2 else raw


def prefecture_on_date(prefecture, date):
    """Prefecture in force on the judgment date (T19: 莱芜市 is part of 济南市 from 2019-01-09)."""
    if prefecture in MERGERS and date is not None and not pd.isna(date):
        new, since = MERGERS[prefecture]
        if pd.Timestamp(date) >= pd.Timestamp(since):
            return new
    return prefecture


@lru_cache(maxsize=None)
def province_of_prefecture(prefecture):
    """Province of a (2016) prefecture name, for courts whose province is missing (T19)."""
    if not isinstance(prefecture, str):
        return None
    if prefecture in MUNI_PREFIX.values():
        return prefecture
    idx, _ = _load()
    provs = {r["province"] for r in idx.get(prefecture, []) if r["prefecture"] == prefecture}
    return provs.pop() if len(provs) == 1 else None


def _isna(x):
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def text_vs_court(text, court_province, court_prefecture, kind="origin", date=None):
    """'1' / '0' / 'NA' for a place text against the court (T19: the one rule for every derived indicator).

    kind 'origin' -> nonlocal (1 = another prefecture); kind 'residence' -> local (1 = same prefecture).
    Address text is never an origin; a missing court province is taken from the court prefecture; prefectures
    are compared as they stood on the judgment date (莱芜 -> 济南 from 2019-01-09)."""
    if kind == "origin":
        text = origin_text_clean(text)
    if _isna(text) or text == "NA" or not text:
        return "NA"
    if _isna(court_province):
        court_province = province_of_prefecture(court_prefecture)
    court_prefecture = prefecture_on_date(None if _isna(court_prefecture) else court_prefecture, date)
    p, f, _, _ = resolve_place(text, court_province, court_prefecture)
    f = prefecture_on_date(f, date)
    if _isna(p) or _isna(court_province):
        return "NA"
    if p != court_province:
        other = True
    elif _isna(f) or _isna(court_prefecture):
        return "NA"
    else:
        other = f != court_prefecture
    return ("1" if other else "0") if kind == "origin" else ("0" if other else "1")


@lru_cache(maxsize=None)
def pref_code_of(prefecture_name=None, prefecture_code=None, province=None):
    """2016 prefecture code for a court: from a (possibly old) prefecture code, or a name."""
    idx, _ = _load()
    if province in MUNI_PREFIX.values():
        return {v: k for k, v in MUNI_PREFIX.items()}[province] + "0000"
    try:
        code = str(int(float(prefecture_code))) if prefecture_code is not None else None
    except ValueError:
        code = None
    if code:
        for r in idx.get(prefecture_name, []):
            if r["code"] == code and r["prefecture_code"]:
                return r["prefecture_code"]
    for r in _prefer_window(idx.get(prefecture_name, [])):
        if r["level"] == "prefecture" or (r["level"] == "county" and r["prefecture"] == r["full_name"]):
            return r["prefecture_code"]
    return None


def pref_name_of_code(code):
    idx, _ = _load()
    for recs in idx.values():
        for r in recs:
            if r["prefecture_code"] == code:
                return r["prefecture"]
    return None


@lru_cache(maxsize=None)
def _code_index():
    idx, _ = _load()
    by_code, code2name = {}, {}
    for recs in idx.values():
        for r in recs:
            if r["prefecture_code"]:
                by_code.setdefault(r["code"], r["prefecture_code"])
                code2name.setdefault(r["prefecture_code"], r["prefecture"])
    return by_code, code2name


@lru_cache(maxsize=None)
def court_prefecture(court_city=None, prefecture_code=None, province=None, court_name=None):
    """Standardised (2016) prefecture name of a court.

    Order: municipality -> meta prefecture name/code -> code alone -> parse the court name.
    """
    by_code, code2name = _code_index()
    if province in MUNI_PREFIX.values():
        return province
    city = None if court_city in (None, "unresolved") else court_city
    code = pref_code_of(city, prefecture_code, province) if (city or prefecture_code) else None
    if code is None and prefecture_code is not None:
        try:
            code = by_code.get(str(int(float(prefecture_code))))
        except ValueError:
            code = None
    if code:
        return code2name.get(code)
    if isinstance(court_name, str):
        p, f, _, st = resolve_place(court_name, province)
        if f and st in ("resolved", "ambiguous_resolved_by_court_prov"):
            return f
    return None
