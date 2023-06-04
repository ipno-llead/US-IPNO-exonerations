model := ../cnn/output/transcript_model
index := ../../thumbnail_accordian/output/index_s1.csv

.PHONY: all clean

all: output/index_transcripts.csv

output/index_transcripts.csv: src/model.py $(input)
	mkdir -p output
	python $< \
		--model $(model) \
		--index $(index) \
		--output $@
