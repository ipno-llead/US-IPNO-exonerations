input := ../classifier/hueristic-filter/output/hueristic-df.csv

.PHONY: all clean

all: output/hueristic-index.csv

output/hueristic-index.csv: src/generate_thumbnails.py $(input)
	-mkdir output
	python3 $< \
		--input=$(input) \
		--output=$@