# FBLeWM paper — ICLR 2027

Official style files from
[iclr-2027-style-files.zip](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).
Do not edit `iclr2027_conference.sty`.

| Date | Item |
|------|------|
| 18 Sep 2026 AOE | Abstract deadline (author list locked) |
| 25 Sep 2026 AOE | Full paper (9 pages main text) |
| | References, AI statement, ethics, reproducibility, appendix: unlimited |

Submission is **double-blind**. Keep `\iclrfinalcopy` commented until camera-ready.

Compile `main.tex` on Overleaf (not `iclr2027_official_sample.tex`).

```bash
cd /home/yuanben/WorldModel/FBLeWM/paper
# Overleaf: pdflatex + bibtex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Number lookup: `../outputs/figures/FIGURES.md`.

## Overleaf git

```bash
cd /home/yuanben/WorldModel/FBLeWM/paper
git add -A
git commit -m "Switch to ICLR 2027 template"
git remote add overleaf https://git.overleaf.com/<PROJECT_ID>
git push -u overleaf master
```

## Locked claim

F/B supply long-range ranking for a 25-step CEM. Do not claim \(F^k(p)\approx z_{t+k}\).
PushT success PNG is CEM=2 at offset 25; main tables are CEM=5.
