"""T18: the second-coder package leaks nothing; agreement scoring on a synthetic example."""
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
C2 = ROOT / "annotation" / "second_coder"


@pytest.mark.skipif(not (C2 / "coder2_sheet.xlsx").exists(), reason="package not built")
def test_package_has_no_prefilled_or_final_values():
    d = pd.read_excel(C2 / "coder2_sheet.xlsx", dtype=str)
    assert list(d.columns) == ["编号", "法院", "判决日期", "origin_place", "residence_place", "status_at_judgment",
                               "ever_bail", "probation", "note"]
    assert len(d) == 50 and d.编号.str.match(r"c\d\d$").all()
    h = (C2 / "viewer_coder2.html").read_text(encoding="utf-8")
    assert "<mark" not in h and not re.search(r"预填|prefill|_final|ai_quote|doc_id", h)
    z = ROOT / "annotation" / "second_coder.zip"           # git-ignored distribution copy; may be absent
    if not z.exists():
        return
    names = zipfile.ZipFile(z).namelist()
    assert sorted(names) == sorted(f"second_coder/{f}" for f in ("coder2_sheet.xlsx", "viewer_coder2.html", "说明.md"))


def test_score_synthetic():
    import T18_coder2_agreement as A
    rows = []
    for i, (o, r, st, eb, pr) in enumerate([("河南省睢县", "同上", "在押", "0", "0"),
                                            ("NA", "北京市朝阳区", "不明", "1", "1"),
                                            ("贵州省荔波县", "NA", "取保", "1", "1")]):
        rows.append({"cid": f"c0{i}", "origin_place": o, "residence_place": r, "status_at_judgment": st,
                     "ever_bail": eb, "probation": pr, "origin_text_final": o, "nonlocal_origin_final": "NA",
                     "residence_local_final": "NA", "status_at_judgment_final": "NA" if st == "不明" else st,
                     "ever_bail_final": eb, "probation_final": pr, "court_province": None, "court_prefecture": None})
    res, per = A.score(pd.DataFrame(rows))
    r = res.set_index("field")
    assert r.at["status_at_judgment", "agreement"] == 1.0      # 不明 == NA
    assert r.at["probation", "agreement"] == 1.0 and r.at["origin_place", "agreement"] == 1.0
