## Makefile for ACD IEEE paper
## Usage:
##   make          -- build PDF (full cycle)
##   make clean    -- remove auxiliary files
##   make distclean -- remove auxiliary files and PDF

MAIN   = paper
LATEX  = pdflatex
BIBTEX = bibtex
FLAGS  = -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib \
             sections/abstract.tex \
             sections/introduction.tex \
             sections/background.tex \
             sections/methodology.tex \
             sections/experimental_setup.tex \
             sections/results.tex \
             sections/discussion.tex \
             sections/conclusion.tex \
             sections/appendix.tex \
             figures/architecture.tex \
             figures/efe_pipeline.tex \
             figures/circuit_flow.tex
	$(LATEX) $(FLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg \
	       $(MAIN).log $(MAIN).out $(MAIN).toc \
	       $(MAIN).lot $(MAIN).lof

distclean: clean
	rm -f $(MAIN).pdf
