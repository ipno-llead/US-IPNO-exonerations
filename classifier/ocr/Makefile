index := ../../index-files/output/text.csv

.PHONY: all clean

all: output/testimony-pdf.csv

output/testimony-pdf.csv: src/ocr.py $(index)
	-mkdir output
	-mkdir -p output/txt
	python $< \
		--index=$(index) \
		--txtdir=output/txt \
		--dpi=300 \
		--output=$@
