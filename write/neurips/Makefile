# vim: set ts=8 sts=0 sw=8 si fenc=utf-8 noet:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:

.PHONY: all clean

all: output/exonerations.pdf

clean: 
	-rm -r output/*

output/%.pdf: src/%.tex src/%.bib src/iclr2024_conference.sty
	-mkdir output
	cd src && pdflatex $*.tex
	cd src && bibtex $*
	cd src && pdflatex $*
	cd src && pdflatex $*
	mv src/$*.pdf output/$*.pdf
	rm src/$*.aux \
		src/$*.log \
		src/$*.out \
		src/$*.bbl \
		src/$*.blg
