labeled := ../../thumbnail_accordian/output/hueristic-index.csv
nonlabeled := ../../thumbnail_accordian/output/index.csv

.PHONY: all clean

all: output/report_model

output/report_model: src/model.py $(input)
	mkdir -p output
	python $< \
		--labeled $(labeled) \
		--nonlabeled $(nonlabeled) \
		--output $@
