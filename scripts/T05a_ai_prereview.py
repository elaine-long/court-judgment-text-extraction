"""T05a: AI pre-review of the T05 audit workbook with a third model (gemini-2.5-pro on Vertex).

    python scripts/T05a_ai_prereview.py trial      # 5 docs; workbook + viewer copies in data/T05a/trial/
    python scripts/T05a_ai_prereview.py run        # all 210 docs; writes annotation/ (backup first)

Protected columns (_prefill, _final, _evidence, changed, flag, note, minutes) are copied cell by cell and
checked against the backup after writing. The model never sees the prefilled values.
"""
import html
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import ai_review as R  # noqa: E402
from src import llm_extract as L  # noqa: E402
from src import paths  # noqa: E402
from src.extract_rules import normalize  # noqa: E402
from src.io import load_meta, load_texts  # noqa: E402
from src.segment import find_spans  # noqa: E402

ANN = paths.ROOT / "annotation"
WB = ANN / "T05_audit.xlsx"
BACKUP = ANN / "T05_audit_backup_preT05a.xlsx"
OUT = paths.DATA / "T05a"
MODEL = L.MODEL_C
BUDGET_USD = 10.0
N_BLIND = 30
WORKERS = 6
PROTECTED_SUFFIX = ("_prefill", "_final", "_evidence")
PROTECTED = ("uid", "court_name", "judgment_date", "changed", "flag", "note", "minutes")
AI_TAIL = ["ai_disagree", "ai_quote_bad", "ai_rule_violation", "ai_blind"]


# ----------------------------------------------------------------------------- sample & inputs
def sample():
    disp = pd.read_csv(ANN / "_private" / "disputes.csv")[["uid", "doc_id"]].assign(sheet="disputes")
    aud = pd.read_csv(ANN / "_private" / "audit_sample.csv")[["uid", "doc_id"]].assign(sheet="audit")
    rng = np.random.default_rng(paths.SEED)
    blind = set(rng.choice(sorted(aud.uid), N_BLIND, replace=False))
    s = pd.concat([disp, aud], ignore_index=True)
    s["ai_blind"] = s.uid.isin(blind).astype(int)
    return s


def court_geo(doc_ids):
    from T06b_run import rule_values
    r = rule_values(doc_ids)
    return r[["court_province", "court_prefecture"]]


# ----------------------------------------------------------------------------- model calls
def run_model(s, texts, max_retries_parse=2):
    ledger = L.Ledger(budget=BUDGET_USD)
    lock = threading.Lock()
    stats = {"api_calls": 0, "cached": 0, "api_errors": 0, "parse_retries": 0, "failed": 0}
    results = {}

    def one(uid, d):
        src = R.build_input(texts.at[d, "judgment"])
        for attempt in range(max_retries_parse + 1):
            with lock:
                ledger.check()
            rec = L.call_cached(MODEL, R.PROMPT, src, ledger=None)
            with lock:
                if rec["cached"]:
                    stats["cached"] += 1
                else:
                    stats["api_calls"] += 1
                    ledger.spent += rec["cost_usd"]
                    ledger.calls += 1
            if rec["error"]:
                with lock:
                    stats["api_errors"] += 1
                continue
            try:
                vals, quotes, errs = R.parse(rec["raw"])
            except ValueError as e:
                errs, vals, quotes = [f"parse:{e}"], None, None
            if vals is not None and not any(e.startswith(("missing", "invalid")) for e in errs):
                return uid, d, src, vals, quotes, errs, rec
            # malformed reply: drop it from the cache and ask again
            L.cache_path(MODEL, rec["key"]).unlink(missing_ok=True)
            with lock:
                stats["parse_retries"] += 1
        with lock:
            stats["failed"] += 1
        return uid, d, src, vals, quotes, errs, rec

    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(one, u, d) for u, d in zip(s.uid, s.doc_id)]
        for i, f in enumerate(as_completed(futs), 1):
            uid, d, src, vals, quotes, errs, rec = f.result()
            results[uid] = {"doc_id": d, "src": src, "vals": vals, "quotes": quotes, "errs": errs,
                            "usage": rec.get("usage"), "cost": rec["cost_usd"] if not rec["cached"] else 0.0,
                            "finish": (rec.get("usage") or {}).get("finish")}
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done  spent ${ledger.spent:.4f}  {time.time() - t0:.0f}s", flush=True)
    stats["spent_usd"] = ledger.spent
    stats["seconds"] = round(time.time() - t0, 1)
    return results, stats


# ----------------------------------------------------------------------------- comparison
def compare(s, results, prefill, geo):
    rows = []
    for r in s.itertuples():
        res = results[r.uid]
        cp, cf = geo.at[r.doc_id, "court_province"], geo.at[r.doc_id, "court_prefecture"]
        cp = None if pd.isna(cp) else cp
        cf = None if pd.isna(cf) else cf
        ok = res["vals"] is not None
        wbv = R.workbook_values(res["vals"], res["quotes"], cp, cf) if ok else {f: ("NA", "NA") for f in R.WB_FIELDS}
        row = {"uid": r.uid, "doc_id": r.doc_id, "sheet": r.sheet, "ai_blind": r.ai_blind, "ok": ok,
               "residence_text_ai": res["vals"].get("residence_text") if ok else None}
        dis, bad = [], []
        viol = R.rule_violations(res["vals"], res["quotes"]) if ok else []
        for f in R.WB_FIELDS:
            v, q = wbv[f]
            row[f"{f}_ai"], row[f"{f}_ai_quote"] = v, q
            pre = prefill.at[r.uid, f"{f}_prefill"]
            row[f"{f}_prefill"] = pre
            agree = R.agrees(f, v, pre, cp, cf) if ok else True
            row[f"{f}_agree"] = agree
            if not agree and f not in viol:
                dis.append(f)
            qf = R.QUOTE_OF[f]
            if ok and qf == f and not R.quote_ok(res["quotes"].get(qf), res["src"]):
                bad.append(f)
        if ok and not R.quote_ok(res["quotes"].get("residence_text"), res["src"]):
            bad.append("residence_local")
        row["ai_disagree"], row["ai_quote_bad"] = ",".join(dis), ",".join(bad)
        row["ai_rule_violation"] = ",".join(viol)
        row["quote_too_long"] = ",".join(f for f in R.MODEL_FIELDS if ok and R.quote_too_long(res["quotes"].get(f)))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- workbook
def read_prefill(path):
    out = []
    for sh in ("disputes", "audit"):
        out.append(pd.read_excel(path, sheet_name=sh, dtype=str))
    return pd.concat(out, ignore_index=True).set_index("uid")


def _is_ai_col(h):
    return h is not None and (h.endswith("_ai") or h.endswith("_ai_quote") or h in AI_TAIL)


def write_workbook(src_path, dst_path, cmp):
    """Rebuild disputes/audit with AI columns after each *_evidence plus three tail columns.
    Re-running replaces earlier AI columns instead of adding new ones."""
    from copy import copy

    from openpyxl import load_workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.cell_range import CellRange

    wb = load_workbook(src_path)
    by_uid = cmp.set_index("uid")
    yellow = PatternFill("solid", fgColor="FFF2CC", bgColor="FFF2CC")
    orange = PatternFill("solid", fgColor="F8CBAD", bgColor="F8CBAD")
    ai_head = PatternFill("solid", fgColor="DDEBF7")
    for name in ("disputes", "audit"):
        old = wb[name]
        idx = wb.sheetnames.index(name)
        head_old = [c.value for c in old[1]]
        keep = [j for j, h in enumerate(head_old, start=1) if not _is_ai_col(h)]
        # new layout: (kind, old column index or AI header)
        layout = []
        for j in keep:
            h = head_old[j - 1]
            layout.append(("old", j))
            if h.endswith("_evidence"):
                f = h[: -len("_evidence")]
                layout += [("ai", f"{f}_ai"), ("ai", f"{f}_ai_quote")]
        layout += [("ai", h) for h in AI_TAIL]
        new = wb.create_sheet(f"{name}__new", idx)
        colmap = {}
        for jn, (kind, ref) in enumerate(layout, start=1):
            letter = get_column_letter(jn)
            if kind == "old":
                colmap[ref] = jn
                ol = get_column_letter(ref)
                if ol in old.column_dimensions:
                    new.column_dimensions[letter].width = old.column_dimensions[ol].width
                for i in range(1, old.max_row + 1):
                    oc, nc = old.cell(row=i, column=ref), new.cell(row=i, column=jn)
                    nc.value = oc.value
                    if oc.has_style:
                        nc._style = copy(oc._style)
                    if oc.comment:
                        nc.comment = copy(oc.comment)
            else:
                hc = new.cell(row=1, column=jn, value=ref)
                hc.font, hc.fill = Font(bold=True), ai_head
                hc.alignment = Alignment(wrap_text=True, vertical="top")
                new.column_dimensions[letter].width = 22 if ref.endswith("_quote") else (16 if ref in AI_TAIL else 10)
        heads = [c.value for c in new[1]]
        uid_col = heads.index("uid") + 1
        for i in range(2, old.max_row + 1):
            uid = new.cell(row=i, column=uid_col).value
            if uid not in by_uid.index:
                continue
            r = by_uid.loc[uid]
            blind = int(r.ai_blind) == 1
            for jn, h in enumerate(heads, start=1):
                if not _is_ai_col(h):
                    continue
                if h == "ai_blind":
                    v = int(r.ai_blind)
                elif blind:
                    v = None
                else:
                    v = r[h] if h in r.index else None
                    v = None if v in ("", None) or (isinstance(v, float) and np.isnan(v)) else v
                    if h.endswith("_ai") and h[:-3] == "term_months" and v not in (None, "NA"):
                        v = float(v)
                    if h.endswith("_ai") and v is not None and not isinstance(v, float):
                        v = str(v)
                new.cell(row=i, column=jn, value=v)
        # data validations: same rules, columns shifted
        for dv in old.data_validations.dataValidation:
            dv2 = copy(dv)
            rngs = []
            for cr in dv.sqref.ranges:
                assert cr.min_col == cr.max_col, "multi-column validation range"
                c = get_column_letter(colmap[cr.min_col])
                rngs.append(f"{c}{cr.min_row}:{c}{cr.max_row}")
            dv2.sqref = " ".join(rngs)
            new.add_data_validation(dv2)
        new.freeze_panes = old.freeze_panes
        # conditional formatting: orange disagreeing *_ai cells first (higher priority), then the yellow row
        n = old.max_row
        last = get_column_letter(len(heads))
        dcol = get_column_letter(heads.index("ai_disagree") + 1)
        for jn, h in enumerate(heads, start=1):
            if h.endswith("_ai"):
                f, c = h[:-3], get_column_letter(jn)
                new.conditional_formatting.add(
                    f"{c}2:{c}{n}", FormulaRule(formula=[f'ISNUMBER(SEARCH(",{f},",","&${dcol}2&","))'],
                                                fill=orange, stopIfTrue=True))
        new.conditional_formatting.add(f"A2:{last}{n}", FormulaRule(formula=[f'LEN(${dcol}2)>0'], fill=yellow))
        wb.remove(old)
        new.title = name
    # keep the original sheet order
    wb._sheets.sort(key=lambda ws: ["disputes", "audit", "codebook", "log"].index(ws.title)
                    if ws.title in ("disputes", "audit", "codebook", "log") else 99)
    wb.active = 0
    wb.save(dst_path)


def verify_protected(ref_path, new_path):
    """Every protected cell in the new workbook equals the reference; returns the number of cells checked."""
    from openpyxl import load_workbook
    a, b = load_workbook(ref_path), load_workbook(new_path)
    n = 0
    for sh in ("disputes", "audit"):
        wa, wb_ = a[sh], b[sh]
        ha = [c.value for c in wa[1]]
        hb = [c.value for c in wb_[1]]
        for j, h in enumerate(ha, start=1):
            if h is None or _is_ai_col(h) or not (h in PROTECTED or h.endswith(PROTECTED_SUFFIX)):
                continue
            jb = hb.index(h) + 1
            for i in range(1, wa.max_row + 1):
                va, vb = wa.cell(row=i, column=j).value, wb_.cell(row=i, column=jb).value
                assert va == vb, f"{sh}!{h} row {i}: {va!r} != {vb!r}"
                n += 1
        assert wa.max_row == wb_.max_row, f"{sh}: row count changed"
        assert len(wa.data_validations.dataValidation) == len(wb_.data_validations.dataValidation)
    for sh in ("codebook", "log"):
        assert [[c.value for c in r] for r in a[sh].iter_rows()] == [[c.value for c in r] for r in b[sh].iter_rows()]
    return n


# ----------------------------------------------------------------------------- viewer
VIEWER_EXTRA_CSS = (":root{--ai:#2563eb;--aibg:#e0ecff}"
                    "@media (prefers-color-scheme:dark){:root{--ai:#7aa7ff;--aibg:#1c2b4a}}"
                    "span.aiq{background:var(--aibg);text-decoration:underline 2px var(--ai);text-underline-offset:3px}"
                    "mark.aiq{text-decoration:underline 2px var(--ai);text-underline-offset:3px}"
                    ".ai{font-size:14px;margin-top:2px}.ai b{color:#c2410c}")


def _spans(t, needles):
    flags = np.zeros(len(t), dtype=bool)
    for nd in needles:
        if not nd or len(nd) < 2:
            continue
        start = 0
        while True:
            k = t.find(nd, start)
            if k < 0:
                break
            flags[k:k + len(nd)] = True
            start = k + 1
    return flags


def _render(seg, pf, af):
    out, i = [], 0
    while i < len(seg):
        j = i
        while j < len(seg) and pf[j] == pf[i] and af[j] == af[i]:
            j += 1
        s = html.escape(seg[i:j])
        if pf[i] and af[i]:
            s = f'<mark class="aiq">{s}</mark>'
        elif pf[i]:
            s = f"<mark>{s}</mark>"
        elif af[i]:
            s = f'<span class="aiq">{s}</span>'
        out.append(s)
        i = j
    return "".join(out)


def build_viewer(path, s, cmp, prefill, texts, meta):
    from T04_build_annotation import SEG_TITLES, VIEWER
    by_uid = cmp.set_index("uid") if cmp is not None else None
    docs = []
    for r in s.itertuples():
        d = r.doc_id
        t = normalize(texts.at[d, "judgment"])
        pre_ev = [prefill.at[r.uid, f"{f}_evidence"] for f in R.WB_FIELDS]
        pre_ev = [e for e in pre_ev if isinstance(e, str)]
        ai_q, ai_line = [], ""
        if r.ai_blind:
            ai_line = "盲审行：本份不显示 AI 提示。"
        elif by_uid is not None and r.uid in by_uid.index:
            row = by_uid.loc[r.uid]
            viol = [v for v in row.ai_rule_violation.split(",") if v]
            ai_q = [row[f"{f}_ai_quote"] for f in R.WB_FIELDS if isinstance(row[f"{f}_ai_quote"], str)
                    and row[f"{f}_ai_quote"] != "NA" and f not in viol]
            ai_q = [sp for q in ai_q if (sp := R.find_quote(q, t))]
            ai_line = ("AI 与预填不一致：<b>" + html.escape(row.ai_disagree.replace(",", "、")) + "</b>"
                       if row.ai_disagree else "AI 与预填一致（仍需核对原文）")
            if viol:
                ai_line += "；AI 违反口径，忽略此提示：<b>" + html.escape("、".join(viol)) + "</b>"
        pf, af = _spans(t, pre_ev), np.zeros(len(t), dtype=bool)
        for a, b in ai_q:
            af[a:b] = True
        sp = find_spans(t)
        segs = sorted((v[0], v[1], k) for k, v in sp.items() if v)
        blocks, pos = [], 0
        for a, b, k in segs or [(0, len(t), None)]:
            if a > pos:
                blocks.append(("其他", pos, a))
            blocks.append((SEG_TITLES.get(k, "全文"), max(a, pos), b))
            pos = max(pos, b)
        if pos < len(t):
            blocks.append(("其他", pos, len(t)))
        out = [(title, _render(t[a:b], pf[a:b], af[a:b])) for title, a, b in blocks]
        docs.append({"id": r.uid, "court": meta.at[d, "court_name"] or "", "date": texts.at[d, "judgment_date"] or "",
                     "cn": meta.at[d, "case_number"] or "", "ai": ai_line, "blocks": out})
    data = json.dumps(docs, ensure_ascii=False).replace("</", "<\\/")
    page = (VIEWER.replace("__TITLE__", "审计阅读器 · T05").replace("__IDNAME__", "编号")
            .replace("浅色底只是关键词提示，不代表任何判断。",
                     "黄色底：预填依据的原文位置；蓝色底加下划线：AI 摘录。两者都只是提示，请回到全文核对。")
            .replace('<div class="meta" id="meta"></div>', '<div class="meta" id="meta"></div><div class="ai" id="ai"></div>')
            .replace("document.getElementById('meta').textContent=d.court+'　'+d.date+'　'+d.cn;",
                     "document.getElementById('meta').textContent=d.court+'　'+d.date+'　'+d.cn;"
                     "document.getElementById('ai').innerHTML=d.ai||'';")
            .replace("</style>", VIEWER_EXTRA_CSS + "</style>")
            .replace("__DATA__", data))
    assert 'id="ai"' in page and "getElementById('ai')" in page, "viewer template changed"
    path.write_text(page, encoding="utf-8")


# ----------------------------------------------------------------------------- summaries
def summary(cmp, stats):
    lines = []
    for sh in ("disputes", "audit"):
        c = cmp[(cmp.sheet == sh) & cmp.ok]
        agree = {f: c[f"{f}_agree"].mean() for f in R.WB_FIELDS}
        bad = {f: c.ai_quote_bad.str.split(",").apply(lambda xs: f in xs).mean() for f in R.WB_FIELDS}
        lines.append(pd.DataFrame({"sheet": sh, "field": R.WB_FIELDS, "n": len(c),
                                   "agree_rate": [agree[f] for f in R.WB_FIELDS],
                                   "quote_bad_rate": [bad[f] for f in R.WB_FIELDS]}))
    tab = pd.concat(lines, ignore_index=True)
    rows = {"n_docs": len(cmp), "n_ok": int(cmp.ok.sum()),
            "yellow_disputes": int(((cmp.sheet == "disputes") & (cmp.ai_disagree != "")).sum()),
            "yellow_audit_nonblind": int(((cmp.sheet == "audit") & (cmp.ai_blind == 0) & (cmp.ai_disagree != "")).sum()),
            "docs_with_quote_bad": int((cmp.ai_quote_bad != "").sum()),
            "docs_with_quote_too_long": int((cmp.quote_too_long != "").sum()),
            "docs_with_rule_violation": int((cmp.ai_rule_violation != "").sum()),
            "rule_violation_fields": cmp.ai_rule_violation[cmp.ai_rule_violation != ""].str.split(",").explode()
                                     .value_counts().to_dict(), **stats}
    return tab, rows


def main(mode):
    t0 = time.time()
    s = sample()
    if mode == "trial":
        # 2 disputes + 3 audit, one of them blind, to exercise every write-back path
        aud = s[s.sheet == "audit"]
        s = pd.concat([s[s.sheet == "disputes"].head(2), aud[aud.ai_blind == 0].head(2), aud[aud.ai_blind == 1].head(1)])
    print(f"mode={mode} model={MODEL} config={L.GEN_CONFIG[MODEL]} docs={len(s)} budget=${BUDGET_USD}")
    ids = s.doc_id.tolist()
    texts = load_texts(ids, columns=["judgment", "judgment_date"]).set_index("doc_id")
    meta = load_meta(["doc_id", "court_name", "case_number"]).set_index("doc_id")
    geo = court_geo(ids)
    prefill = read_prefill(BACKUP if BACKUP.exists() else WB)

    results, stats = run_model(s, texts)
    cmp = compare(s, results, prefill, geo)
    tab, rows = summary(cmp, stats)
    print(json.dumps(rows, ensure_ascii=False, indent=1))

    out = OUT / ("trial" if mode == "trial" else "")
    out.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(out / "ai_results.csv", index=False)
    tab.to_csv(out / "agreement_by_field.csv", index=False)
    (out / "run_stats.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame([{"uid": u, "doc_id": r["doc_id"], "errs": ";".join(r["errs"]), "finish": r["finish"],
                   "usage": json.dumps(r["usage"]), "cost": r["cost"]} for u, r in results.items()]) \
        .to_csv(out / "calls.csv", index=False)

    if mode == "trial":
        src = BACKUP if BACKUP.exists() else WB
        wb_out, viewer_out = out / "T05_audit_trial.xlsx", out / "viewer_audit_trial.html"
    else:
        if not BACKUP.exists():
            shutil.copy2(WB, BACKUP)
            print(f"backup -> {BACKUP.name}")
        src, wb_out, viewer_out = WB, WB, ANN / "viewer_audit.html"
        blind = cmp[cmp.ai_blind == 1]
        blind.to_csv(ANN / "_private" / "ai_blind_results.csv", index=False)
    tmp = out / "_tmp_workbook.xlsx"
    write_workbook(src, tmp, cmp)
    n = verify_protected(BACKUP if BACKUP.exists() else WB, tmp)
    shutil.move(tmp, wb_out)
    print(f"protected cells identical: {n}")
    full = sample()
    if mode == "trial":
        full = full[full.uid.isin(s.uid)]
    all_texts = texts if mode == "trial" else load_texts(full.doc_id.tolist(), columns=["judgment", "judgment_date"]).set_index("doc_id")
    build_viewer(viewer_out, full, cmp, prefill, all_texts, meta)
    print(f"wrote {wb_out.relative_to(paths.ROOT)}, {viewer_out.relative_to(paths.ROOT)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trial")
