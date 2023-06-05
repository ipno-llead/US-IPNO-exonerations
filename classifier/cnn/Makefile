labeled := ../../thumbnail_accordian/output/hueristic-index.csv
nonlabeled := ../../thumbnail_accordian/output/index.csv

.PHONY: all clean

all: output/transcript_model

output/transcript_model: src/model.py $(input)
	mkdir -p output
	python $< \
		--labeled $(labeled) \
		--nonlabeled $(nonlabeled) \
		--output $@
