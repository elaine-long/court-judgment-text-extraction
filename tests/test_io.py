import time

import numpy as np
import pyarrow.parquet as pq
import pytest

from src import paths
from src.case_number import parse_doc_kind, parse_prov_code
from src.io import load_meta, load_texts

CASES = [
    # case_number, abbr, court_code, doc_kind
    ("(2018)粤0902刑初370号", "粤", "0902", "刑初"),
    ("(2017)豫1282刑初571号", "豫", "1282", "刑初"),
    ("(2018)浙07刑终21号", "浙", "07", "刑终"),
    ("(2019)京0105刑更123号", "京", "0105", "刑更"),
    ("（2019）沪0115刑初1号", "沪", "0115", "刑初"),
    ("(2016)津0116刑初字第8号", "津", "0116", "刑初"),
    ("(2016)鲁0683行初19号", "鲁", "0683", "其他"),
    ("(2015)兴刑初字第153号", None, None, "刑初"),
    ("(2014)惠东法刑二初字第108号", None, None, "刑初"),
    ("(2015)武法民初字第01659号", None, None, "其他"),
    ("(2018)内0102刑初88号", "内", "0102", "刑初"),
    ("(2018)川01刑监5号", "川", "01", "其他"),
    ("永刑初字第70号", None, None, "刑初"),
    ("2015年裕刑初字第00302号", None, None, "刑初"),
]


@pytest.mark.parametrize("cn,abbr,code,kind", CASES)
def test_case_number_parsing(cn, abbr, code, kind):
    assert parse_prov_code(cn) == (abbr, code)
    assert parse_doc_kind(cn) == kind


needs_data = pytest.mark.skipif(
    not (paths.META.exists() and paths.DOCS_TEXT.exists()), reason="working data not built")




@needs_data
def test_meta_read_speed():
    t = time.time()
    m = load_meta()
    assert time.time() - t < 5
    assert len(m) > 0


