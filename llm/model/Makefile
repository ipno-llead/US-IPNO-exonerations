CSV_PATH := ../../ocr/output/reports.csv
OUTPUT_DIR := output
PYTHON_SCRIPT := src/model.py

all: process_all

process_all:
	python3 $(PYTHON_SCRIPT) --csv_path $(CSV_PATH) --output_dir $(OUTPUT_DIR)

clean:
	find $(OUTPUT_DIR) -name '*.json' -exec rm {} +
	find $(OUTPUT_DIR) -name '*.csv' -exec rm {} +

.PHONY: all process_all clean
