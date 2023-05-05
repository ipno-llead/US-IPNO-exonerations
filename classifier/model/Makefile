input := ../../thumbnail_accordian/output/accordian.csv

.PHONY: all clean

all: output/rf_transcripts_model.h5

output/rf_transcripts_model.h5: src/model.py $(input)
	mkdir -p output
	python $< \
		--input $(input) \
		--output $@
