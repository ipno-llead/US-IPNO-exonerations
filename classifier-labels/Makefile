index := ../../index-files/output/test_testimony.csv

.PHONY: all clean

all: output/hueristic-df.csv

output/hueristic-df.csv: src/hueristics.py $(index)
	-mkdir output
	python $< \
		--index=$(index) \
		--output=$@
