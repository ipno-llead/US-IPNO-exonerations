# vim: set ts=8 sts=0 sw=8 si fenc=utf-8 noet:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:

.PHONY: all clean symlinks

all: symlinks

clean: 
	-rm -r output/*

# note: we force clean out old symlinks to avoid issues where we remove a file
# from the input repo but carry stale symlinks through the import pipeline
# doesn't change the mod date of the resulting "file" for make purposes, so
# this is free to do
symlinks: src/symlinks.py clean
	-mkdir output
	python $< --inputdir=input --outputdir=output

# done.
