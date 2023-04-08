index := ../hueristic_filter/output/hueristic-df.csv

.PHONY: all clean

all: output/hueristic-pdfs.csv

output/hueristic-pdfs.csv: src/ocr.py $(index)
	-mkdir output
	-mkdir -p output/txt
	python $< \
		--index=$(index) \
		--txtdir=output/txt \
		--dpi=300 \
		--output=$@
