"""T05a: AI pre-review of the T05 audit sample with a third model (outside the extraction pipeline).

The model sees the full judgment (never the prefilled values), returns a value and a verbatim quote for
11 fields; nonlocal_origin / residence_local are derived from its place names with the gazetteer.
"""
import re

import pandas as pd

from src import llm_extract as L
from src.extract_rules import normalize
from src.gazetteer import resolve_place
from src.segment import find_spans

PROMPT = "audit_review_v2"   # v1 (2026-10-05, archived) inferred procedure / custody against the codebook
MAX_CHARS = 12000
QUOTE_MAX = 60
# fields the model returns
MODEL_FIELDS = {
    "origin_text": "text", "residence_text": "text",
    "status_at_judgment": {"在押", "取保", "监视居住", "其他", "NA"},
    "ever_bail": {"1", "0"}, "plea_formal": {"1", "0"}, "jiejie": {"1", "0"},
    "procedure": {"速裁", "简易", "普通", "NA"}, "counsel_any": {"1", "0"},
    "penalty_type": {"有期徒刑", "拘役", "管制", "单处罚金", "免予刑事处罚", "其他"},
    "term_months": "number", "probation": {"1", "0"},
}
# the 12 workbook fields, in workbook order; derived fields take the quote of the place-name field
WB_FIELDS = ["origin_text", "nonlocal_origin", "residence_local", "status_at_judgment", "ever_bail",
             "plea_formal", "jiejie", "procedure", "counsel_any", "penalty_type", "term_months", "probation"]
SAME_AS = re.compile(r"(?:住址|住所)?同(?:上|前|户籍(?:地|所在地)?)")
QUOTE_OF = {f: f for f in WB_FIELDS} | {"nonlocal_origin": "origin_text", "residence_local": "residence_text"}


def build_input(text):
    """Full normalised text; above MAX_CHARS keep only the party, reasoning and verdict sections."""
    t = normalize(text)
    if len(t) <= MAX_CHARS:
        return t
    sp = dict(find_spans(t))
    if sp.get("header"):
        nxt = [v[0] for k, v in sp.items() if v and k != "header" and v[0] >= sp["header"][1]]
        sp["header"] = (0, min(nxt) if nxt else len(t))
    parts = [f"【{lab}】\n{t[sp[k][0]:sp[k][1]]}" for lab, k in
             (("当事人段", "header"), ("本院认为段", "reasoning"), ("判决段", "verdict")) if sp.get(k)]
    return "\n\n".join(parts) if parts else t[:MAX_CHARS]


def parse(raw):
    """Return ({field: value}, {field: quote}, [errors]). Invalid or missing values become None."""
    obj = L.parse_json(raw)
    vals, quotes, errs = {}, {}, []
    for f, allowed in MODEL_FIELDS.items():
        item = obj.get(f)
        if item is None:
            errs.append(f"missing:{f}")
            vals[f], quotes[f] = None, "NA"
            continue
        v, q = (item.get("value"), item.get("quote")) if isinstance(item, dict) else (item, None)
        v = L._norm_value(v)
        if allowed == "number":
            if v != "NA":
                try:
                    v = f"{float(re.sub(r'[^0-9.]', '', v)):g}"
                except ValueError:
                    errs.append(f"invalid:{f}")
                    v = None
        elif allowed != "text" and v not in allowed:
            errs.append(f"invalid:{f}")
            v = None
        vals[f] = v
        quotes[f] = L._norm_value(q)
    return vals, quotes, errs


def _quote_regex(quote):
    """Each quote character may also match a run of U+FFFD: the source has encoding damage in places and the
    model restores the character (e.g. source "河南���睢县", quote "河南省睢县"). Whitespace is ignored."""
    q = re.sub(r"\s+", "", quote)
    return re.compile(r"\s*".join(f"(?:{re.escape(c)}|\ufffd+)" for c in q))


def find_quote(quote, source):
    """(start, end) of the first match of the quote in the source, or None."""
    if quote in (None, "NA", ""):
        return None
    if quote in source:
        k = source.index(quote)
        return k, k + len(quote)
    m = _quote_regex(quote).search(source)
    return m.span() if m else None


def quote_ok(quote, source):
    """True when the quote is NA/empty or a verbatim substring of the text sent to the model. Length is not
    checked here (an over-long but verbatim quote is not fabricated); see quote_too_long."""
    return quote in (None, "NA", "") or find_quote(quote, source) is not None


def quote_too_long(quote):
    return quote not in (None, "NA", "") and len(quote) > QUOTE_MAX


def _migrant(p, f, court_prov, court_pref):
    if p is None or not court_prov:
        return None
    if p != court_prov:
        return True
    if f is None or court_pref is None:
        return None
    return f != court_pref


def geo(text, court_prov, court_pref, kind, date=None):
    """Same rule as the pipeline (T19: delegates to gazetteer.text_vs_court)."""
    from src.gazetteer import text_vs_court
    return text_vs_court(text, court_prov, court_pref, "origin" if kind == "origin" else "residence", date)


def workbook_values(vals, quotes, court_prov, court_pref):
    """{field: (value, quote)} for the 12 workbook fields."""
    out = {f: (vals.get(f) if vals.get(f) is not None else "NA", quotes.get(QUOTE_OF[f], "NA"))
           for f in WB_FIELDS if f in MODEL_FIELDS}
    out["nonlocal_origin"] = (geo(vals.get("origin_text"), court_prov, court_pref, "origin"), quotes.get("origin_text", "NA"))
    res = vals.get("residence_text")
    if isinstance(res, str) and SAME_AS.fullmatch(res):  # "住址同上" -> the place named before it
        res = vals.get("origin_text")
    out["residence_local"] = (geo(res, court_prov, court_pref, "res"), quotes.get("residence_text", "NA"))
    return {f: out[f] for f in WB_FIELDS}


def _same_place(a, b, court_prov, court_pref):
    if a == b:
        return True
    if "NA" in (a, b):
        return False
    if a in b or b in a:
        return True
    pa, fa, _, _ = resolve_place(a, court_prov, court_pref)
    pb, fb, _, _ = resolve_place(b, court_prov, court_pref)
    return pa is not None and fa is not None and (pa, fa) == (pb, fb)


def agrees(field, ai, prefill, court_prov=None, court_pref=None):
    """AI value vs prefilled value. origin_text: same string, one contains the other, or same prefecture."""
    a = "NA" if ai in (None, "") else str(ai).strip()
    b = "NA" if prefill in (None, "") or (isinstance(prefill, float) and pd.isna(prefill)) else str(prefill).strip()
    if field == "origin_text":
        return _same_place(a, b, court_prov, court_pref)
    if field == "term_months" and "NA" not in (a, b):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except ValueError:
            return False
    return a == b


# ----------------------------------------------------------------------------- codebook violations
# Checked on the model's own quote. A violating field is shown in the workbook but not counted as a
# disagreement; the viewer tells the auditor to ignore that hint.
_PROC_BASIS = re.compile(r"程序|[（(]速[）)]|刑速")
_OFFSET = re.compile(r"折抵|即自[^。；]{0,30}起至|刑期自[^。；]{0,30}起至|自[^。；]{0,20}起至[^。；]{0,20}止")
_CUSTODY = re.compile(r"羁押于|押于|看守所|逮捕|拘留|服刑|监狱|取保|监视居住|在家|抓获|归案|投案")


def rule_violations(vals, quotes):
    """Fields whose AI value rests on a basis the codebook forbids.

    procedure: a non-NA value whose quote has no explicit "……程序" and no speedy case-number marker
               (v1 inferred 普通 from "依法组成合议庭").
    status_at_judgment: a non-NA value whose quote is the sentence-offset clause ("即自……起至……止",
               "折抵") with no custody event in it.
    """
    out = []
    v, q = vals.get("procedure"), quotes.get("procedure") or "NA"
    if v not in (None, "NA") and not _PROC_BASIS.search(q):
        out.append("procedure")
    v, q = vals.get("status_at_judgment"), quotes.get("status_at_judgment") or "NA"
    if v not in (None, "NA") and (q == "NA" or (_OFFSET.search(q) and not _CUSTODY.search(q))):
        out.append("status_at_judgment")
    return out
