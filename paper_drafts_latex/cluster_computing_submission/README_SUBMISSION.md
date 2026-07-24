# Cluster Computing (Springer) — DIO submission package

**Journal:** [Cluster Computing](https://link.springer.com/journal/10586)  
**Guidelines:** https://link.springer.com/journal/10586/submission-guidelines  
**Submit:** Use “Submit manuscript” on the journal home page (SNAPP / Editorial Manager).

## Files

| File | Role |
|------|------|
| `DIO_ClusterComputing.tex` | **Main manuscript** (single TeX file — do not `\input` other sources) |
| `DIO_ClusterComputing.pdf` | Compiled PDF for review |
| `sn-bibliography.bib` | BibTeX database |
| `sn-jnl.cls` + `*.bst` | Springer Nature template (from official package) |
| `*.png` | Figures (also upload separately if portal asks) |
| `cuted.sty` / `stfloats.sty` | Local stubs if TeX install lacks `sttools` |

## Build

```bash
pdflatex DIO_ClusterComputing
bibtex DIO_ClusterComputing
pdflatex DIO_ClusterComputing
pdflatex DIO_ClusterComputing
```

Journal recommends `sn-jnl` with **`[iicol]`** when your TeX has the full `sttools` package (`cuted.sty`).  
This build uses single-column content-first layout if `iicol` is unavailable; Springer production reformats to journal style.

## Guidelines checklist (completed in MS)

- [x] LaTeX using Springer Nature `sn-jnl` template  
- [x] Title, authors, affiliations, emails  
- [x] Abstract + keywords  
- [x] Decimal headings (≤3 levels)  
- [x] Numbered references (`sn-mathphys-num`)  
- [x] Figures with captions  
- [x] Tables with captions  
- [x] Algorithm environment  
- [x] **Declarations** (funding, COI, data, code, author contribution, ethics)  
- [x] Code availability (GitHub)  
- [x] Multi-SKU **placeholder table** (Section multi-SKU) for your upcoming runs  
- [x] Honest dual-T4 multi-seed results already filled  

## Before final submit

1. Fill **Table multiSKU** after multi-SKU GPU tests.  
2. Optionally enable `[iicol]` in `\documentclass` when `cuted.sty` is installed.  
3. Choose **subscription / non-OA** at acceptance for **$0 APC**.  
4. Optional: post arXiv preprint and cite the arXiv ID in the cover letter.  
5. Upload: main `.tex`, `.bib`, figures, compiled PDF, and any SI.

## Cover letter sketch

> Dear Editor,  
> We submit “DIO: Dual-Timescale Predictive Orchestration for Heterogeneous LLM Inference Clusters” for consideration as a Research Article. DIO is an open-source cluster control plane that ranks LLM backends with dual-timescale NLMS and joint cost routing over stock vLLM. Multi-seed dual-T4 results and open software are included; multi-SKU physical validation is in progress. We choose the subscription track (no APC).  

## Do not submit yet if

- You want multi-SKU numbers in the first version → wait and fill Table multiSKU.  
- Or submit now with TBD multi-SKU and update at revision (honest approach).
