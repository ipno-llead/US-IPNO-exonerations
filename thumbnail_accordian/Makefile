input := ../index-files/output/index.csv

.PHONY: all clean

all: output/index.csv

output/index.csv: src/accordian.py $(input)
	-mkdir output
	python $< \
		--input=$(input) \
		--output=$@