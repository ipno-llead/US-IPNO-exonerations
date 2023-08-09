txt_dir := input/transcripts
csv_dir := output/transcripts

inputs := $(shell find $(txt_dir) -name '*.txt')
outputs := $(patsubst $(txt_dir)/%.txt,$(csv_dir)/%.csv,$(inputs))

script := src/model.py

.PHONY: all clean

all: $(outputs)

$(csv_dir)/%.csv: $(script) $(txt_dir)/%.txt
	-mkdir -p $(shell dirname $@)
	python $< \
		--input=$(txt_dir)/$*.txt \
		--output=$@

#all: $(txt_dir)
#	mkdir -p $(csv_dir)
#	find $(txt_dir) -name "*.txt" | while read file; do \
#		python $(script) --input "$$file" --output "$(csv_dir)/$$(basename "$$file" .txt).csv"; \
#	done

clean:
	rm -rf $(csv_dir)
