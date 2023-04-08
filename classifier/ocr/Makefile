index := ../../index-files/output/test.csv

.PHONY: all clean

all: output/testimony-pdfs.csv

output/testimony-pdfs.csv: src/ocr.py $(index)
	-mkdir output
	-mkdir -p output/txt
	python $< \
		--index=$(index) \
		--txtdir=output/txt \
		--dpi=300 \
		--output=$@
