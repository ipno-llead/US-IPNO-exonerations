txt_dir := input/transcripts
csv_dir := output/transcripts

script := src/model.py

.PHONY: all clean

all: $(txt_dir)
	mkdir -p $(csv_dir)
	find $(txt_dir) -name "*.txt" | while read file; do \
		python $(script) --input "$$file" --output "$(csv_dir)/$$(basename "$$file" .txt).csv"; \
	done

clean:
	rm -rf $(csv_dir)