INPUT_PATH_TRANSCRIPTS := input/transcripts
INPUT_PATH_REPORTS := input/reports
OUTPUT_PATH_TRANSCRIPTS := output/transcripts
OUTPUT_PATH_REPORTS := output/reports

PYTHON_SCRIPT := src/model.py

all: process_all

process_all:
	python3 $(PYTHON_SCRIPT) --input_path_transcripts $(INPUT_PATH_TRANSCRIPTS) --input_path_reports $(INPUT_PATH_REPORTS) --output_path_transcripts $(OUTPUT_PATH_TRANSCRIPTS) --output_path_reports $(OUTPUT_PATH_REPORTS)

clean:
	rm -rf $(OUTPUT_PATH_TRANSCRIPTS)/*
	rm -rf $(OUTPUT_PATH_REPORTS)/*

.PHONY: all process_all clean
