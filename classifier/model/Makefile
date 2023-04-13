input := ../ocr/output/hueristic-pdfs.csv

.PHONY: all clean

all: output/padme.h5

output/padme.h5: src/model.py $(input)
	mkdir -p output
	python $< \
		--input $(input) \
		--output $@
