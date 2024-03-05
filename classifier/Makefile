labeled := ../../thumbnails/output/hueristic-index.csv
nonlabeled := ../../thumbnails/output/index.csv

.PHONY: all clean

all: output/transcript_model

output/transcript_model: src/model.py $(input)
	mkdir -p output
	python3 $< \
		--labeled $(labeled) \
		--nonlabeled $(nonlabeled) \
		--output $@
