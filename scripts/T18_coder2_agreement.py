"""T18 §5: agreement between the second coder and the first coder's final values.

    python scripts/T18_coder2_agreement.py                    # annotation/second_coder/coder2_sheet.xlsx
    python scripts/T18_coder2_agreement.py --fake             # build and score a FAKE sheet (data/T18/), no real coder
    python scripts/T18_coder2_agreement.py --sheet PATH --tag NAME

Reads only the *_final columns of annotation/T05_audit.xlsx (sheet "audit"). origin_place / residence_place are
turned into nonlocal_origin / residence_local with the same gazetteer rule as the pipeline (court prefecture).
Reports, per field: agreement rate, Cohen's kappa (categorical fields), and 95% bootstrap CIs (2,000 draws over
judgments). Disagreeing judgment ids go to annotation/_private/coder2_disagreements*.csv (data/T18/ for --fake).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import paths  # noqa: E402
from src.ai_review import SAME_AS, _same_place, geo  # noqa: E402

ANN = paths.ROOT / "annotation"
FAKE_DIR = paths.DATA / "T18"
REPS = 2000
FIELDS = ["origin_place", "nonlocal_origin", "residence_local", "status_at_judgment", "ever_bail", "probation"]


def _s(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "NA", "nan", "None", "不明"):
        return "NA"
    s = str(v).strip()
    return s[0] if s in ("1.0", "0.0") else s


def load(sheet):
    m = pd.read_csv(ANN / "_private" / "coder2_map.csv")
    c2 = pd.read_excel(sheet, dtype=str, keep_default_na=False).rename(columns={"编号": "cid"})
    fin_cols = ["uid", "origin_text_final", "nonlocal_origin_final", "residence_local_final",
                "status_at_judgment_final", "ever_bail_final", "probation_final"]
    a = pd.read_excel(ANN / "T05_audit.xlsx", sheet_name="audit", dtype=str, keep_default_na=False, usecols=fin_cols)
    df = m.merge(c2, on="cid").merge(a, on="uid")
    from T06b_run import rule_values
    R = rule_values(df.doc_id.tolist())[["court_province", "court_prefecture"]]
    return df.join(R, on="doc_id")


def score(df):
    rows, per = [], []
    for r in df.itertuples():
        cp = None if pd.isna(r.court_province) else r.court_province
        cf = None if pd.isna(r.court_prefecture) else r.court_prefecture
        o2, res2 = _s(r.origin_place), _s(r.residence_place)
        if SAME_AS.fullmatch(res2):
            res2 = o2
        c2 = {"origin_place": o2, "nonlocal_origin": geo(o2, cp, cf, "origin"), "residence_local": geo(res2, cp, cf, "res"),
              "status_at_judgment": _s(r.status_at_judgment), "ever_bail": _s(r.ever_bail), "probation": _s(r.probation)}
        c1 = {"origin_place": _s(r.origin_text_final), "nonlocal_origin": _s(r.nonlocal_origin_final),
              "residence_local": _s(r.residence_local_final), "status_at_judgment": _s(r.status_at_judgment_final),
              "ever_bail": _s(r.ever_bail_final), "probation": _s(r.probation_final)}
        for f in FIELDS:
            same = _same_place(c2[f], c1[f], cp, cf) if f == "origin_place" else c2[f] == c1[f]
            per.append({"cid": r.cid, "field": f, "c1": c1[f], "c2": c2[f], "agree": int(same)})
    per = pd.DataFrame(per)
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(paths.SEED)
    cids = per.cid.unique()
    for f in FIELDS:
        g = per[per.field == f].set_index("cid")
        kap = np.nan if f == "origin_place" or g.c1.nunique() + g.c2.nunique() <= 2 and g.c1.nunique() == 1 \
            else cohen_kappa_score(g.c1, g.c2)
        boots_a, boots_k = [], []
        for _ in range(REPS):
            s = g.loc[rng.choice(cids, len(cids))]
            boots_a.append(s.agree.mean())
            if f != "origin_place" and s.c1.nunique() > 1:
                boots_k.append(cohen_kappa_score(s.c1, s.c2))
        rows.append({"field": f, "n": len(g), "agreement": g.agree.mean(),
                     "agree_ci_low": np.quantile(boots_a, 0.025), "agree_ci_high": np.quantile(boots_a, 0.975),
                     "kappa": kap, "kappa_ci_low": np.quantile(boots_k, 0.025) if boots_k else np.nan,
                     "kappa_ci_high": np.quantile(boots_k, 0.975) if boots_k else np.nan,
                     "derived": f in ("nonlocal_origin", "residence_local")})
    return pd.DataFrame(rows), per


def make_fake(path, noise=0.1):
    """A FAKE second-coder sheet from the hybrid prefill and rule residence text, with random errors.
    Never uses the first coder's values."""
    from openpyxl import Workbook
    from T06b_post import prefill_table
    from T06b_run import rule_values
    m = pd.read_csv(ANN / "_private" / "coder2_map.csv")
    ids = m.doc_id.tolist()
    pre, _ = prefill_table(ids)
    R = rule_values(ids)
    rng = np.random.default_rng(paths.SEED + 18)
    wb = Workbook()
    ws = wb.active
    ws.append(["编号", "法院", "判决日期", "origin_place", "residence_place", "status_at_judgment", "ever_bail", "probation", "note"])
    for r in m.itertuples():
        st = pre[r.doc_id]["status_at_judgment"]
        st = "不明" if st == "NA" else st
        row = [pre[r.doc_id]["origin_text"], R.at[r.doc_id, "residence_text"], st,
               pre[r.doc_id]["ever_bail"], pre[r.doc_id]["probation"]]
        if rng.random() < noise:
            row[2] = rng.choice(["在押", "取保", "不明"])
        if rng.random() < noise:
            row[0] = "NA"
        ws.append([r.cid, "", ""] + row + ["FAKE"])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(ANN / "second_coder" / "coder2_sheet.xlsx"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    if a.fake:
        sheet, tag = FAKE_DIR / "fake_coder2_sheet.xlsx", "_fake"
        make_fake(sheet)
        dis_dir, out_dir = FAKE_DIR, FAKE_DIR
    else:
        sheet, tag = Path(a.sheet), a.tag
        dis_dir, out_dir = ANN / "_private", paths.TABLES
    df = load(sheet)
    if (df[["origin_place", "status_at_judgment", "ever_bail", "probation"]] == "").all(axis=None):
        sys.exit("the second-coder sheet is still empty")
    res, per = score(df)
    res.to_csv(out_dir / f"T18_coder2_agreement{tag}.csv", index=False)
    per[per.agree == 0][["cid", "field"]].to_csv(dis_dir / f"coder2_disagreements{tag}.csv", index=False)
    print(res.round(3).to_string(index=False))
    print(f"disagreeing field cells: {int((per.agree == 0).sum())} -> {dis_dir}")


if __name__ == "__main__":
    main()
