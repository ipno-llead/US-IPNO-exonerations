input := ../index-files/output/index.csv

.PHONY: all clean

all: output/index.csv

output/index.csv: src/generate_thumbnails.py $(input)
	-mkdir output
	python3 $< \
		--input=$(input) \
		--output=$@