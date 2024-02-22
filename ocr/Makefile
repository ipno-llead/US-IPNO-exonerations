index := ../classifier/hueristic-filter/output/hueristic-df.csv

.PHONY: all clean

all: output/hueristic-reports

output/hueristic-reports: src/ocr.py $(index)
	-mkdir -p output/json
	python3 $< \
		--index=$(index) \
		--jsondir=output/json \
		--dpi=300 \
		--output=output/hueristic-reports.csv

clean:
	-rm -rf output/json
	-rm output/hueristic-reports.csv
