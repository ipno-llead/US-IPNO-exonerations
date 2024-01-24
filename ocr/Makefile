index := ../classifier/hueristic-filter/output/hueristic-df.csv

.PHONY: all clean

all: output/hueristic-transcripts-json

output/hueristic-transcripts-json: src/ocr.py $(index)
	-mkdir -p output/json
	python3 $< \
		--index=$(index) \
		--jsondir=output/json \
		--dpi=300 \
		--output=output/hueristic-transcripts-json.csv

clean:
	-rm -rf output/json
	-rm output/hueristic-transcripts-json.csv
