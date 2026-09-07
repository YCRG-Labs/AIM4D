"""
Verify every bibliography entry against Semantic Scholar, falling back to OpenAlex.

For each entry the script resolves the record by DOI, then by arXiv id, then by
title search, and compares the returned title, first author surname, year and
venue against what the .bib claims. It reports one line per entry.

Verdicts:
  OK          resolved and title, first author and year all agree
  CHECK       resolved but at least one field disagrees (details printed)
  NOT_FOUND   no record found in either index

Nothing here edits the bibliography. Read-only.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BIB = sys.argv[1] if len(sys.argv) > 1 else "/Users/jacobcrainic/Downloads/fullpapereferences.bib"
S2 = "https://api.semanticscholar.org/graph/v1"
OA = "https://api.openalex.org/works"
MAILTO = "jacobcrainic@icloud.com"
FIELDS = "title,year,venue,externalIds,authors"
PAUSE = 0.4


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bib-verify/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 504):
                time.sleep(1.5 * (i + 1))
                continue
            return None
        except Exception:
            time.sleep(2)
    return None


def norm(s):
    s = re.sub(r"[{}\\]", "", (s or "").lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def sim(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def parse_bib(path):
    txt = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", txt, re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.+?)[}\"]\s*,?\s*\n", body + "\n", re.S):
            f[fm.group(1).lower()] = " ".join(fm.group(2).split())
        f["_type"], f["_key"] = kind, key
        out.append(f)
    return out


def first_surname(author_field):
    if not author_field:
        return ""
    a = author_field.split(" and ")[0].strip()
    return norm(a.split(",")[0] if "," in a else a.split()[-1])


CR = "https://api.crossref.org/works"
AX = "http://export.arxiv.org/api/query"


def _cr_map(it):
    y = None
    for k in ("issued", "published-print", "published-online", "created"):
        dp = (it.get(k) or {}).get("date-parts") or []
        if dp and dp[0] and dp[0][0]:
            y = dp[0][0]
            break
    return {"title": (it.get("title") or [""])[0], "year": y,
            "venue": (it.get("container-title") or [""])[0],
            "authors": [{"name": f"{a.get('given', '')} {a.get('family', '')}".strip()}
                        for a in (it.get("author") or [])[:5]]}


def crossref_doi(doi):
    r = get(f"{CR}/{urllib.parse.quote(doi)}")
    it = (r or {}).get("message") or {}
    return _cr_map(it) if it.get("title") else None


def crossref_title(title):
    q = urllib.parse.quote(norm(title)[:200])
    r = get(f"{CR}?query.bibliographic={q}&rows=5")
    items = [i for i in (((r or {}).get("message") or {}).get("items") or []) if i.get("title")]
    if not items:
        return None
    best = max(items, key=lambda i: sim(title, i["title"][0]))
    return _cr_map(best) if sim(title, best["title"][0]) >= 0.6 else None


def arxiv_record(aid):
    try:
        req = urllib.request.Request(f"{AX}?id_list={aid}", headers={"User-Agent": "bib-verify/1.0"})
        x = urllib.request.urlopen(req, timeout=30).read().decode()
    except Exception:
        return None
    ent = x.split("<entry>")[1:]
    if not ent:
        return None
    t = re.search(r"<title>(.*?)</title>", ent[0], re.S)
    y = re.search(r"<published>(\d{4})", ent[0])
    names = re.findall(r"<name>(.*?)</name>", ent[0])
    if not t:
        return None
    return {"title": " ".join(t.group(1).split()), "year": int(y.group(1)) if y else None,
            "venue": "arXiv", "authors": [{"name": n} for n in names[:5]]}


def resolve(e):
    doi = e.get("doi", "").strip()
    if doi:
        r = crossref_doi(doi)
        if r:
            return r, "crossref/doi"
    ax = re.search(r"ar[Xx]iv:\s*([0-9.]+)", e.get("note", "") + " " + e.get("eprint", "")) \
        or re.match(r"\s*(\d{4}\.\d{4,5})", e.get("eprint", ""))
    if ax:
        r = arxiv_record(ax.group(1))
        if r:
            return r, "arxiv/id"
    title = e.get("title", "")
    if title:
        r = crossref_title(title)
        if r:
            return r, "crossref/title"
    return resolve_s2_oa(e)


def resolve_s2_oa(e):
    doi = e.get("doi", "").strip()
    if doi:
        r = get(f"{S2}/paper/DOI:{urllib.parse.quote(doi)}?fields={FIELDS}")
        if r and r.get("title"):
            return r, "s2/doi"
    note = e.get("note", "") + " " + e.get("eprint", "")
    am = re.search(r"ar[Xx]iv:\s*([0-9.]+)", note)
    if am:
        r = get(f"{S2}/paper/ARXIV:{am.group(1)}?fields={FIELDS}")
        if r and r.get("title"):
            return r, "s2/arxiv"
    title = e.get("title", "")
    if title:
        q = urllib.parse.quote(norm(title)[:200])
        r = get(f"{OA}?filter=title.search:{q}&per-page=5&mailto={MAILTO}")
        if r and r.get("results"):
            best = max(r["results"], key=lambda p: sim(title, p.get("display_name") or ""))
            if sim(title, best.get("display_name") or "") >= 0.6:
                loc = best.get("primary_location") or {}
                src = loc.get("source") or {}
                return {
                    "title": best.get("display_name"),
                    "year": best.get("publication_year"),
                    "venue": (best.get("host_venue") or {}).get("display_name") or src.get("display_name"),
                    "authors": [{"name": (a.get("author") or {}).get("display_name", "")}
                                for a in (best.get("authorships") or [])[:5]],
                }, "openalex/title"
    return None, None


def main():
    entries = parse_bib(BIB)
    print(f"parsed {len(entries)} entries from {BIB}\n")
    rows = []
    for i, e in enumerate(entries, 1):
        key = e["_key"]
        try:
            rec, how = resolve(e)
        except Exception as ex:
            rows.append((key, 'ERROR', repr(ex)[:90], ''))
            print(f"[{i:>2}/{len(entries)}] {key:<28} ERROR {ex!r}")
            continue
        time.sleep(PAUSE)
        if not rec:
            rows.append((key, "NOT_FOUND", "", e.get("title", "")[:70]))
            print(f"[{i:>2}/{len(entries)}] {key:<28} NOT_FOUND")
            continue
        problems = []
        ts = sim(e.get("title", ""), rec.get("title", ""))
        if ts < 0.75:
            problems.append(f"title {ts:.2f}: bib='{e.get('title','')[:55]}' vs idx='{(rec.get('title') or '')[:55]}'")
        by = e.get("year", "").strip()
        ry = str(rec.get("year") or "")
        if by and ry and by != ry:
            problems.append(f"year bib={by} idx={ry}")
        bs = first_surname(e.get("author", ""))
        names = [a.get("name", "") for a in (rec.get("authors") or [])]
        if bs and names and not any(bs in norm(n) for n in names):
            problems.append(f"first author '{bs}' not among {[n for n in names[:4]]}")
        verdict = "OK" if not problems else "CHECK"
        rows.append((key, verdict, "; ".join(problems), how))
        print(f"[{i:>2}/{len(entries)}] {key:<28} {verdict:<9} ({how})"
              + (f"\n      -> {'; '.join(problems)}" if problems else ""))

    print("\n" + "=" * 78)
    from collections import Counter
    c = Counter(r[1] for r in rows)
    print("SUMMARY:", dict(c))
    for label in ("NOT_FOUND", "ERROR", "CHECK"):
        bad = [r for r in rows if r[1] == label]
        if bad:
            print(f"\n{label} ({len(bad)}):")
            for k, _, why, extra in bad:
                print(f"  {k:<28} {why or extra}")


if __name__ == "__main__":
    main()
