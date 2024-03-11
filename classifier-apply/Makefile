model := ../classifier/output/transcript_model
index := ../thumbnails/output/index.csv

.PHONY: all clean

all: output/index_transcripts.csv

output/index_transcripts.csv: src/model.py $(input)
	mkdir -p output
	python3 $< \
		--model $(model) \
		--index $(index) \
		--output $@
