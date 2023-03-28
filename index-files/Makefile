docs_path := input/wrongful-convictions-docs
out_path := output/index.csv


.PHONY: all clean

all: $(out_path)

$(out_path): src/index.py
	-mkdir output
	python $< \
		--path="$(docs_path)" \
		--outdir="$(out_path)"
