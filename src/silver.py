"""T09e1b: the "silver" label on the validation sample V.

For each (document, field) in data/V_labels.parquet:
- rule and model A agree (no adjudication)  -> silver = that value;
- they disagree -> model B adjudicated (B_adj_side): 'rule' -> rule value, 'A' -> model A value,
  'third' -> B's own value. For nonlocal_origin / residence_local the third value is place text, turned into
  1/0/NA with the same gazetteer rule as T06b (court prefecture of the document).
bail_at_judgment = 1 if status is 取保, 0 for any other status, NA if status is NA (as in the panel).

Caveat: where rule and A agree the silver label equals them, so errors shared by the rule and model A are
invisible; the human audit measures the silver label's own accuracy (T09e2).
"""
import numpy as np
import pandas as pd

FIELDS = ["nonlocal_origin", "residence_local", "status_at_judgment", "ever_bail", "penalty_type", "term_months",
          "probation"]


def _s(v, field=None):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "NA"
    v = str(v)
    if field == "term_months" and v != "NA":   # "18.0" and "18" are the same value
        try:
            return f"{float(v):g}"
        except ValueError:
            return v
    return v


def silver_labels(vl, court=None):
    """vl: V_labels rows; court: DataFrame indexed by doc_id with court_province, court_prefecture (needed for
    'third' place texts). Returns a wide DataFrame indexed by doc_id with one column per field + bail_at_judgment
    and n_adjudicated."""
    from src.ai_review import geo
    rows = {}
    for r in vl[vl.field.isin(FIELDS)].itertuples(index=False):
        side = r.B_adj_side if isinstance(r.B_adj_side, str) else None
        if side is None:
            val = _s(r.rule, r.field) if _s(r.rule, r.field) == _s(r.A, r.field) else "NA"
        elif side == "rule":
            val = _s(r.rule, r.field)
        elif side == "A":
            val = _s(r.A, r.field)
        else:
            val = _s(r.B_adj_value)
            if r.field in ("nonlocal_origin", "residence_local") and val not in ("1", "0", "NA"):
                cp = court.at[r.doc_id, "court_province"] if court is not None and r.doc_id in court.index else None
                cf = court.at[r.doc_id, "court_prefecture"] if court is not None and r.doc_id in court.index else None
                cp = None if cp is None or pd.isna(cp) else cp
                cf = None if cf is None or pd.isna(cf) else cf
                val = geo(val, cp, cf, "origin" if r.field == "nonlocal_origin" else "res")
        if r.field == "term_months" and val != "NA":
            val = f"{float(val):g}"
        d = rows.setdefault(r.doc_id, {"n_adjudicated": 0})
        d[r.field] = val
        d["n_adjudicated"] += side is not None
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "doc_id"
    if "status_at_judgment" in out:
        st = out["status_at_judgment"].fillna("NA")
        out["bail_at_judgment"] = np.where(st == "NA", "NA", np.where(st == "取保", "1", "0"))
    return out
