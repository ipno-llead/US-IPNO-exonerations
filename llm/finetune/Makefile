key := input/key.txt
training_data := input/training-data.jsonl

.PHONY: all clean

all: run

run: src/finetune.py
	python3 src/finetune.py --input "$(training_data)" --apikey "$(key)"

clean:
	@echo "Add commands to clean up any generated files"
