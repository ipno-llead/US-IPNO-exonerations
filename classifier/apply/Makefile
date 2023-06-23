model := ../cnn/output/report_model
index := ../../thumbnail_accordian/output/index.csv

.PHONY: all clean

all: output/index_reports.csv

output/index_reports.csv: src/model.py $(input)
	mkdir -p output
	python $< \
		--model $(model) \
		--index $(index) \
		--output $@
