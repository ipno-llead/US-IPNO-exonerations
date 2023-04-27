input := ../classifier/hueristic-filter/output/hueristic-df.csv

.PHONY: all clean

all: output/accordian.csv

output/accordian.csv: src/accordian.py $(input)
	-mkdir output
	python $< \
		--input=$(input) \
		--output=$@