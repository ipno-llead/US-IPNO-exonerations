model := ../cnn/output/report_model
index := ../../thumbnail_accordian/output/index_v1.csv

.PHONY: all clean

all: output/report_transcripts.csv

output/report_transcripts.csv: src/model.py $(input)
	mkdir -p output
	python $< \
		--model $(model) \
		--index $(index) \
		--output $@
