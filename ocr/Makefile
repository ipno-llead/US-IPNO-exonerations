index := ../classifier/hueristic-filter/output/hueristic-df.csv

.PHONY: all clean

all: output/hueristic-transcripts-txt.csv

output/hueristic-transcripts-txt.csv: src/ocr.py $(index)
	-mkdir output
	-mkdir -p output/txt
	python $< \
		--index=$(index) \
		--txtdir=output/txt \
		--dpi=300 \
		--output=$@
