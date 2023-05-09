labeled := ../../thumbnail_accordian/output/accordian.csv
nonlabeled := ../../thumbnail_accordian/output/index.csv

.PHONY: all clean

all: output/rf_reports_model

output/rf_reports_model: src/model.py $(input)
	mkdir -p output
	python $< \
		--labeled $(labeled) \
		--nonlabeled $(nonlabeled) \
		--output $@
