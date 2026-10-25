"""T19: place-resolution (gazetteer) fixes for the migrant variables.

    python scripts/T19_geo_fix.py diagnose   # §0: classify the own-city "migrant" cases, counts by pilot x period
    python scripts/T19_geo_fix.py compare    # §2: old vs new migrant_city_nores / migrant_city / local_residence

Case list: annotation/_private/T19_owncity_migrant_cases.csv (origin text contains the court city's name, coded
migrant). No LLM.
"""
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import paths  # noqa: E402

T, D = paths.TABLES, paths.DATA
CASES = paths.ROOT / "annotation" / "_private" / "T19_owncity_migrant_cases.csv"
AGENCY = re.compile(r"公安局|分局|派出所|看守所|检察院|法院|拘留所|监狱")
RESID = re.compile(r"居住|住址|现住|暂住|住所|家住")
CATS = ["4 court province missing", "2 agency name in origin text", "5 Laiwu district (Jinan, 2019)",
        "1a municipality + unresolved rest -> pre-1997 Sichuan record", "1b new district (2016-2019) without 2016 code",
        "3 county substring overrides explicit city", "6 address words in origin text (correct: origin elsewhere)",
        "0 other / false positive"]


def _case_frame():
    c = pd.read_csv(CASES)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.register("c", c[["doc_id"]])
    d = con.execute(f"""SELECT r.doc_id, r.hukou_raw, r.birthplace_raw, r.native_raw, r.origin_prefecture, r.origin_county,
            r.migrant_city_nores, f.court_prefecture_g, t.court_province_final, t.judgment_date_parsed AS jdate
        FROM read_parquet('{D / "defendants_rules.parquet"}') r
        JOIN read_parquet('{D / "feat_rules.parquet"}') f USING (doc_id)
        JOIN read_parquet('{D / "treatment.parquet"}') t USING (doc_id)
        WHERE r.def_idx = 0 AND r.doc_id IN (SELECT doc_id FROM c)""").df()
    d["src_raw"] = d[["hukou_raw", "birthplace_raw", "native_raw"]].bfill(axis=1).iloc[:, 0]
    return c.merge(d, on="doc_id", how="left")


def classify(r, now_prov):
    raw = r.origin_raw if isinstance(r.origin_raw, str) else ""
    city = r.city if isinstance(r.city, str) else ""
    city_short = city[:-1] if city.endswith("市") else city
    if pd.isna(r.court_province_final):
        return CATS[0]
    if AGENCY.search(raw):
        return CATS[1]
    if city == "济南市" and "莱芜" in raw:
        return CATS[2]
    if RESID.search(raw):
        return CATS[6]
    if city in ("重庆市", "北京市", "天津市", "上海市") and now_prov != city:
        return CATS[3]
    if city_short and city_short in raw:
        if isinstance(r.oc, str) and pd.isna(r.op):
            return CATS[4]
        if isinstance(r.oc, str) and r.op != city:
            return CATS[5]
    return CATS[7]


def diagnose():
    from src.extract_rules import _migrant
    from src.gazetteer import resolve_place
    c = _case_frame()
    res = [resolve_place(r.origin_raw, None if pd.isna(r.court_province_final) else r.court_province_final,
                         r.court_prefecture_g) for r in c.itertuples()]
    c["now_prov"], c["now_pref"], c["now_county"] = [x[0] for x in res], [x[1] for x in res], [x[2] for x in res]
    c["category"] = [classify(r, pv) for r, pv in zip(c.itertuples(), c.now_prov)]
    c["now_migrant"] = [_migrant(p, f, cp, cf, "city") for (p, f, _, _), cp, cf in
                        zip(res, c.court_province_final, c.court_prefecture_g)]
    c.to_csv(paths.ROOT / "annotation" / "_private" / "T19_cases_classified.csv", index=False)
    tab = c.pivot_table(index="category", columns=["pilot", "period"], values="doc_id", aggfunc="count", fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab = tab.reindex([x for x in CATS if x in tab.index])
    tab.columns = [f"pilot{p}_{q}" if q else str(p) for p, q in tab.columns]
    tab.to_csv(T / "T19_case_categories.csv")
    print(tab.to_string())
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 30)
    for cat in tab.index:
        g = c[c.category == cat]
        print(f"\n== {cat}: {len(g)}")
        print(g[["doc_id", "city", "year", "origin_raw", "op", "oc", "now_pref", "now_county"]].head(6).to_string(index=False))


GEO_FIELDS = ("nonlocal_origin", "residence_local")


def refresh_frozen():
    """Recompute only the geography-derived values (nonlocal_origin, residence_local) of the frozen LLM-layer tables
    from the cached model responses (data/llm/T06b/calls_*.csv; no API calls) and the rebuilt rules. Every other
    column is unchanged. Originals: data/archive_T18/."""
    import T06b_run as R6
    from src import llm_extract as L
    # V_labels
    vl = pd.read_parquet(D / "V_labels.parquet")
    ids = vl.doc_id.unique().tolist()
    RV = R6.rule_values(ids)
    A = R6.parse_records(R6._calls("V_A"), L.FIELDS_V2).set_index("doc_id")
    B = R6.parse_records(R6._calls("V_B"), L.FIELDS_V2).set_index("doc_id")
    av = {d: R6.model_values(A, d, RV.loc[d]) for d in ids if d in A.index and d in RV.index and A.at[d, "status"] != "call_error"}
    bv = {d: R6.model_values(B, d, RV.loc[d]) for d in ids if d in B.index and d in RV.index and B.at[d, "status"] != "call_error"}
    m = vl.field.isin(GEO_FIELDS)
    old = vl.loc[m, ["rule", "A", "B"]].copy()
    vl.loc[m, "rule"] = [RV.at[d, f] if d in RV.index else None for d, f in zip(vl.doc_id[m], vl.field[m])]
    vl.loc[m, "A"] = [av.get(d, {}).get(f) for d, f in zip(vl.doc_id[m], vl.field[m])]
    vl.loc[m, "B"] = [bv.get(d, {}).get(f) for d, f in zip(vl.doc_id[m], vl.field[m])]
    for c in ("rule", "A", "B"):
        vl[c] = vl[c].astype("string")
    ch_v = {c: int((old[c].fillna("NA").astype(str) != vl.loc[m, c].fillna("NA").astype(str)).sum()) for c in ("rule", "A", "B")}
    vl.to_parquet(D / "V_labels.parquet", index=False)
    # R_fill
    rf = pd.read_parquet(D / "R_fill.parquet")
    need_ids = rf.loc[rf.field.isin(GEO_FIELDS), "doc_id"].unique().tolist()
    RR = R6.rule_values(need_ids)
    AR = R6.parse_records(R6._calls("R_A"), L.FIELDS_V2).set_index("doc_id")
    BR = R6.parse_records(R6._calls("R_B"), L.FIELDS_V2).set_index("doc_id")
    m = rf.field.isin(GEO_FIELDS)
    old = rf.loc[m, ["rule", "A", "B", "filled", "filled_by"]].copy()
    for i in rf.index[m]:
        d, f = rf.at[i, "doc_id"], rf.at[i, "field"]
        if d not in RR.index:
            continue
        r = RR.loc[d]
        a = R6.model_values(AR, d, r) if d in AR.index and AR.at[d, "status"] != "call_error" else None
        b = R6.model_values(BR, d, r) if d in BR.index and BR.at[d, "status"] != "call_error" else None
        x = a.get(f) if a else None
        y = b.get(f) if b else None
        ok = x not in (None, "NA") and y not in (None, "NA") and not pd.isna(x) and not pd.isna(y) and R6.agree(f, x, y)
        rf.at[i, "rule"], rf.at[i, "A"], rf.at[i, "B"] = r[f], x, y
        rf.at[i, "filled"] = x if ok else None
        rf.at[i, "filled_by"] = "A=B" if ok else ("llm_missing" if not (a and b) else
                                                  "both_NA" if (x in (None, "NA") or pd.isna(x)) and (y in (None, "NA") or pd.isna(y))
                                                  else "unresolved_llm")
    for c in ("rule", "A", "B", "filled"):
        rf[c] = rf[c].astype("string")
    ch_r = {c: int((old[c].fillna("NA").astype(str) != rf.loc[m, c].fillna("NA").astype(str)).sum()) for c in old.columns}
    rf.to_parquet(D / "R_fill.parquet", index=False)
    srow = rf.groupby("field").agg(n=("doc_id", "size"), filled=("filled_by", lambda x: (x == "A=B").sum()),
                                   both_NA=("filled_by", lambda x: (x == "both_NA").sum()),
                                   unresolved=("filled_by", lambda x: (x == "unresolved_llm").sum()),
                                   missing=("filled_by", lambda x: (x == "llm_missing").sum())).reset_index()
    srow["fill_rate"] = srow.filled / srow.n
    srow.to_csv(T / "T06b_R_fill_summary.csv", index=False)
    pd.DataFrame([{"table": "V_labels", **ch_v}, {"table": "R_fill", **ch_r}]).to_csv(T / "T19_frozen_refresh_changes.csv", index=False)
    print("V_labels changed cells (geo rows):", ch_v)
    print("R_fill changed cells (geo rows):", ch_r)
    print(srow.round(3).to_string(index=False))



def compare():
    """Change matrices old x new (archive_T18 vs rebuilt) for the migrant variables, by pilot x period."""
    def lab(x):
        return x.map({1.0: "1", 0.0: "0", True: "1", False: "0"}).fillna("NA")
    o = pd.read_parquet(D / "archive_T18" / "analysis_hybrid_full.parquet",
                        columns=["doc_id", "pilot", "period", "migrant", "migrant_city", "local_residence"])
    n = pd.read_parquet(D / "analysis_hybrid_full.parquet",
                        columns=["doc_id", "pilot", "period", "migrant", "migrant_city", "local_residence"])
    m = o.merge(n, on="doc_id", how="outer", suffixes=("_old", "_new"))
    m["pilot"] = m.pilot_new.fillna(m.pilot_old)
    m["period"] = m.period_new.fillna(m.period_old)
    rows = []
    for v in ("migrant", "migrant_city", "local_residence"):
        m[f"{v}_o"], m[f"{v}_n"] = lab(m[f"{v}_old"]), lab(m[f"{v}_new"])
        t = m.groupby(["pilot", "period", f"{v}_o", f"{v}_n"]).size().rename("n").reset_index()
        t = t.rename(columns={f"{v}_o": "old", f"{v}_n": "new"})
        t.insert(0, "variable", v)
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(T / "T19_change_matrix.csv", index=False)
    chg = out[out.old != out.new]
    summ = chg.pivot_table(index=["variable", "old", "new"], columns=["pilot", "period"], values="n", aggfunc="sum",
                           fill_value=0)
    summ["total"] = summ.sum(axis=1)
    print(summ.to_string())
    main_o = pd.read_parquet(D / "archive_T18" / "analysis_hybrid.parquet", columns=["doc_id", "migrant"])
    main_n = pd.read_parquet(D / "analysis_hybrid.parquet", columns=["doc_id", "migrant"])
    print(f"main sample: old {len(main_o):,} (migrant share {main_o.migrant.mean():.4f}); "
          f"new {len(main_n):,} (migrant share {main_n.migrant.mean():.4f}); "
          f"left {len(set(main_o.doc_id) - set(main_n.doc_id)):,}, entered {len(set(main_n.doc_id) - set(main_o.doc_id)):,}")


if __name__ == "__main__":
    t0 = time.time()
    {"diagnose": diagnose, "refresh_frozen": refresh_frozen, "compare": compare}[sys.argv[1]]()
    print(f"{sys.argv[1]} {time.time() - t0:.0f}s")
