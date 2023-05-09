# vim: set ts=8 sts=0 sw=8 si fenc=utf-8 noet:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:
# Authors:     TS
# Maintainers: TS
# Copyright:   2023, HRDAG, GPL v2 or later
# =========================================

.PHONY: all clean

# this file should be a data frame that contains at least the following cols:
#
#   - filename: path to pdf file from current dir
#   - pageno: space separated list of pages to be processed
#             example: "2 3 4 7 9"
#   - outname: name to give the output json file. the json will be written to
#              'output/json/{outname}.json'
pdf_index := path/to/index-file.parquet

# creds is a text file with two lines
#     the first line is the url for the azure service
#     the second line is a valid api key
creds := path/to/creds.txt

all: output/ms.done

clean: 
	-rm -r output/*

output/ms.done: src/ms-prebuilt.py $(index)
	-mkdir -p output/json
	python $< \
		--input=$(index) \
		--creds=$(creds)
	touch $@

# done.
