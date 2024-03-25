index := ../classifier-filter/output/reports.csv

.PHONY: all clean

all: output/reports

output/reports: src/ocr.py $(index)
	-mkdir -p output/json
	python3 $< \
		--index=$(index) \
		--jsondir=output/json \
		--dpi=300 \
		--output=output/reports.csv

clean:
	-rm -rf output/json
	-rm output/reports.csv
