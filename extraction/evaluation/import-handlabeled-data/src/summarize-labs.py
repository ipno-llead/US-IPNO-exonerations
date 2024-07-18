# vim: set ts=4 sts=0 sw=4 si fenc=utf-8 et:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:

import argparse
import pandas as pd
from pathlib import Path


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eachdir', default='output/each')
    parser.add_argument('--output')
    return parser.parse_args()


args = getargs()
labs = pd.concat(pd.read_parquet(f) for f in Path(args.eachdir).rglob('*.parquet'))
labs.columns
params = ['chunk_size', 'chunk_overlap', 'temperature', 'k', 'hyde']

smry = labs.fillna(0) \
        .groupby(params + ['filetype']) \
        .agg({'FN': sum, 'FP': sum, 'TP': sum,
              'file': len}) \
        .rename(columns = {'file': 'n_files'}) \
        .reset_index()

smry['precision'] = smry.TP / (smry.TP + smry.FP)
smry['recall'] = smry.TP / (smry.TP + smry.FN)

smry.sort_values(['filetype', 'precision']).to_excel("output/overall-summary.xlsx", index=False)
