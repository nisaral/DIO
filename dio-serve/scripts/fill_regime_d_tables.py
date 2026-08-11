#!/usr/bin/env python3
"""Fill the Regime D LaTeX tables from a completed run's summary.json.

Typing numbers out of a JSON file into a .tex by hand is how papers end up with
figures that disagree with their tables. This does it mechanically.

    python scripts/fill_regime_d_tables.py \
        --summary results_regime_d/summary.json \
        --tex ../paper_drafts_latex/cluster_computing_submission/DIO_ClusterComputing.tex

Writes in place after saving <tex>.bak. Use --dry-run to see the new table bodies
without touching the file. Re-running is safe: it rewrites the same two table
bodies rather than appending.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

STRAT_LABEL = {
    "nlms": "NLMS",
    "rls": "RLS",
    "least_loaded": "Least-Loaded",
    "round_robin": "RR",
}
# Row order in tab:realD. RR last so the "vs RR" column reads down to its baseline.
STRAT_ORDER = ("nlms", "rls", "least_loaded", "round_robin")


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def ms(stat: dict | None, key: str) -> str:
    """Render a mean/std pair as $mean\\pm std$, or --- when absent."""
    if not isinstance(stat, dict):
        return "---"
    node = stat.get(key)
    if not isinstance(node, dict) or node.get("mean") is None:
        return "---"
    mean, std = node["mean"], node.get("std")
    if std is None:
        return f"${mean:.0f}$"
    return f"${mean:.0f}\\pm{std:.0f}$"


def pct(node: dict | None) -> str:
    if not isinstance(node, dict) or node.get("mean") is None:
        return "---"
    mean, std = node["mean"], node.get("std")
    if std is None:
        return f"${mean:.1f}\\%$"
    return f"${mean:.1f}\\pm{std:.1f}\\%$"


def num(v, digits: int = 3) -> str:
    return "---" if v is None else f"${v:.{digits}f}$"


def build_realD(d1: dict) -> str:
    """Body rows for tab:realD."""
    if not d1:
        die("summary.json has no D1_strategy_head_to_head cell")
    rows = []
    best = None
    for s in STRAT_ORDER:
        cell = d1.get(s)
        if not cell:
            continue
        p99 = cell.get("p99", {}).get("mean")
        if p99 is not None and s != "round_robin":
            if best is None or p99 < best[1]:
                best = (s, p99)
    for s in STRAT_ORDER:
        cell = d1.get(s)
        if not cell:
            continue
        p99_cell = ms(cell, "p99")
        if best and s == best[0]:
            p99_cell = p99_cell.replace("$", "$\\mathbf{", 1).rstrip("$") + "}$"
        # run_d1 stores improvements at the D1 level, not inside each strategy:
        # "p99_improvement_vs_rr" for nlms, "..._vs_rr_<strat>" for the others.
        if s == "round_robin":
            vs = "---"
        else:
            key = ("p99_improvement_vs_rr" if s == "nlms"
                   else f"p99_improvement_vs_rr_{s}")
            imp = d1.get(key)
            vs = pct(imp)
            if isinstance(imp, dict) and imp.get("wins") is not None:
                vs += f" ({imp['wins']}/{imp['of']})"
        rows.append(
            f"{STRAT_LABEL[s]} & {ms(cell, 'p50')} & {ms(cell, 'p95')} & "
            f"{p99_cell} & {vs} \\\\"
        )
    return "\n".join(rows)


def build_slopeD(d2: dict) -> str:
    """Body rows for tab:slopeD, including the ratio row the scheduler consumes."""
    if not d2:
        return (
            "\\multicolumn{5}{c}{D2 cell not present in this run "
            "(re-run with \\texttt{--d2})} \\\\"
        )
    by_worker = d2.get("slopes_by_worker") or {}
    if not by_worker:
        die("D2 cell present but slopes_by_worker is empty")

    rows, learned, ols = [], {}, {}
    for wid in sorted(by_worker):
        w = by_worker[wid]
        lm = (w.get("learned") or {}).get("mean")
        of = w.get("ols") or {}
        osl, r2 = of.get("slope"), of.get("r2")
        vram = w.get("declared_vram_mb")
        if lm is not None:
            learned[wid] = lm
        if osl is not None:
            ols[wid] = osl
        note = of.get("note")
        r2_cell = num(r2) if not note else "\\emph{degenerate}"
        rows.append(
            f"{wid.upper()} & {'---' if vram is None else f'${vram:.0f}$'} & "
            f"{num(lm)} & {num(osl)} & {r2_cell} \\\\"
        )

    rows.append("\\midrule")
    if len(learned) == 2 and len(ols) == 2:
        a, b = sorted(learned)  # alphabetical: a100 before l4
        lr = learned[a] / learned[b] if learned[b] else None
        orr = ols[a] / ols[b] if ols[b] else None
        rows.append(
            f"Ratio ({a.upper()}/{b.upper()}) & --- & {num(lr, 2)} & "
            f"{num(orr, 2)} & --- \\\\"
        )
    elif len(learned) > 2 or len(ols) > 2:
        # Three or more SKUs: one pair cannot describe the fleet, so emit a row
        # per worker normalised against the slowest card. Falls back to computing
        # the normalisation here if the summary predates run_d2 writing it.
        nrm = (d2.get("slope_ratio") or {}).get("normalized") or {}

        def ratios(kind: str, src: dict) -> tuple[str, dict]:
            cell = nrm.get(kind) or {}
            if cell.get("ratios"):
                return cell.get("reference", ""), cell["ratios"]
            usable = {w: v for w, v in src.items() if v}
            if len(usable) < 2:
                return "", {}
            ref = max(usable, key=lambda w: usable[w])
            return ref, {w: usable[w] / usable[ref] for w in usable}

        lref, lrat = ratios("learned", learned)
        oref, orat = ratios("ols", ols)
        ref = lref or oref
        for wid in sorted(set(lrat) | set(orat)):
            rows.append(
                f"\\quad rel. {wid.upper()} & --- & {num(lrat.get(wid), 2)} & "
                f"{num(orat.get(wid), 2)} & --- \\\\"
            )
        if ref:
            rows.append(
                f"\\multicolumn{{5}}{{@{{}}l@{{}}}}{{\\footnotesize Slopes "
                f"normalised against {ref.upper()} (slowest).}} \\\\"
            )
    else:
        rows.append("Ratio & --- & --- & --- & --- \\\\")
    return "\n".join(rows)


def replace_body(tex: str, label: str, body: str) -> str:
    """Swap the rows between \\midrule and \\bottomrule in the table with `label`."""
    i = tex.find(f"\\label{{{label}}}")
    if i < 0:
        die(f"could not find \\label{{{label}}} in the .tex")
    start = tex.find("\\midrule", i)
    end = tex.find("\\bottomrule", start)
    if start < 0 or end < 0:
        die(f"table {label} has no \\midrule/\\bottomrule to fill")
    return tex[: start + len("\\midrule")] + "\n" + body + "\n" + tex[end:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--tex", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.summary.exists():
        die(f"no such summary: {args.summary}")
    if not args.tex.exists():
        die(f"no such tex: {args.tex}")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("mock"):
        die(
            "this summary.json came from a --mock run. Mock numbers must never "
            "reach the paper. Re-run on real GPUs first."
        )

    realD = build_realD(summary.get("D1_strategy_head_to_head") or {})
    slopeD = build_slopeD(summary.get("D2_slope_identifiability") or {})

    print("=== tab:realD ===")
    print(realD)
    print("\n=== tab:slopeD ===")
    print(slopeD)

    if args.dry_run:
        print("\n(dry run — .tex untouched)")
        return 0

    tex = args.tex.read_text(encoding="utf-8")
    tex = replace_body(tex, "tab:realD", realD)
    tex = replace_body(tex, "tab:slopeD", slopeD)

    backup = args.tex.with_suffix(args.tex.suffix + ".bak")
    shutil.copy2(args.tex, backup)
    args.tex.write_text(tex, encoding="utf-8")

    print(f"\nwrote {args.tex}  (backup: {backup.name})")
    remaining = len(re.findall(r"\bTBD\b", tex))
    print(f"TBD markers still in the .tex: {remaining}")
    if remaining:
        print("  -> the prose 'Results. TBD' paragraph still needs writing by hand.")
    print("next: pdflatex + bibtex + pdflatex + pdflatex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
