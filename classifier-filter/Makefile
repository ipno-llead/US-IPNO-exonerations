# Variables for input file paths
HEURISTIC_DF := ../classifier-labels/output/hueristic-df.csv
CLASSIFIER_DF := ../classifier-apply/output/index_reports.csv

# Target output file
OUTPUT := output/reports.csv

# Mark special targets that do not correspond to files
.PHONY: all clean

# Default target
all: $(OUTPUT)

# Rule to create the output file
$(OUTPUT): src/filter.py $(HEURISTIC_DF) $(CLASSIFIER_DF)
	mkdir -p $(dir $@)
	python3 $< --hueristic_df $(HEURISTIC_DF) --classifier_df $(CLASSIFIER_DF) --output $@

# Clean target to remove the generated output
clean:
	rm -f $(OUTPUT)
