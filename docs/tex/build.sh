#!/usr/bin/env bash
set -euo pipefail
name="hardened_pnm"
rm -f $name.{aux,bbl,blg,log,out,pdf}
#rm -f *.png
#ln -s graphs/*.png
pdflatex ${name}.tex
bibtex ${name}
pdflatex ${name}.tex
pdflatex ${name}.tex
echo "Built ${name}.pdf ✅"

