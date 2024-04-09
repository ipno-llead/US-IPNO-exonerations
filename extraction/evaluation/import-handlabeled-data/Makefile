# vim: set ts=8 sts=0 sw=8 si fenc=utf-8 noet:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:

inputdir := ../symlink-handlabeled-data/output
inputs := $(shell find $(inputdir) -type l -iname '*.csv')

eachdir := output/each
eachout := $(patsubst $(inputdir)/%.csv,$(eachdir)/%.parquet,$(inputs))

.PHONY: all clean

all: $(eachout)

clean: 
	-rm -r output/*

$(eachdir)/%.parquet: src/import-labs.py $(inputdir)/%.csv
	-mkdir -p $(shell dirname $@)
	python $< \
		--input=$(inputdir)/$*.csv \
		--output=$@
	
# done.
