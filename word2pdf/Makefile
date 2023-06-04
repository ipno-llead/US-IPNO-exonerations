input := ../index-files/output/word_index.csv

.PHONY: all clean

all: output/word_index.csv

output/word_index.csv: src/word2pdf.py $(input)
	-mkdir output
	python $< \
		--input=$(input) \
		--output=$@