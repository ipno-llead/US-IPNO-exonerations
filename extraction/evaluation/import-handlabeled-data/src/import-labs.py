# vim: set ts=4 sts=0 sw=4 si fenc=utf-8 et:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:
# Authors:     TS
# Maintainers: TS
# Copyright:   2023, HRDAG, GPL v2 or later
# =========================================

import argparse
import pandas as pd
import os

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default = '../make-symlinks/output/labeled-data-batch-01/transcripts/query_k_20/01-trial-transcript.docx.csv')
    parser.add_argument('--output')
    return parser.parse_args()


def clean_name(name):
    return name.lower().replace(" ", "_").replace(".", "_")


# currently, we're only focused on the officer name extraction
def import_labfile(fname):
    labs = pd.read_csv(args.input)
    labs.columns = [clean_name(colname) for colname in labs.columns]
    assert len({'label', 'true_label'}.intersection(labs.columns)) == 1
    assert len({'missing', 'missing_officer'}.intersection(labs.columns)) == 1
    labs = labs.rename(columns = {'true_label': 'label',
                                  'missing': 'missing_officer'})
    out = labs[['officer_name', 'query', 'chunk_size', 'chunk_overlap', \
            'temperature', 'k', 'hyde', 'label', 'missing_officer']] \
            .drop_duplicates()
    #out['label'] = out.label.fillna(0.0)
    #assert not any(out.missing_officer.isna())
    return out


# note: for mentions that were not extracted but should have been, all
# parameter values will be na
def explode_labels(labs, parameters):
    unique_params = labs[parameters].dropna().drop_duplicates()
    all_officers = labs[['officer_name']].drop_duplicates()
    key = all_officers.merge(unique_params, how = 'cross')
    labs_exp = pd.merge(key, labs,
                        on = ['officer_name'] + parameters,
                        how = 'left')
    assert len(labs_exp) == len(key)
    assert all(labs_exp.missing_officer.isna() | labs_exp.label.notna())
    return labs_exp


# label: missing means was not extracted (false negative)
#        1 means correctly extracted (true positive)
#        0 means incorrectly extracted (false positive)
def recode_trilabel(label):
    return label.fillna(-1).replace({0: 'FP', 1: 'TP', -1: 'FN'})


def determine(grp):
    if any(grp.missing_officer > 0):
        return 'FN'
    elif any(grp.label > 0):
        return 'TP'
    else:
        return 'FP'


def doctype(x):
    if 'transcripts' in x:
        return 'transcript'
    elif 'police-reports' in x:
        return 'police-report'
    else:
        raise ValueError("undefined doctype")


if __name__ == '__main__':
    args = getargs()
    params = ['chunk_size', 'chunk_overlap', 'temperature', 'k', 'hyde']
    labs = import_labfile(args.input)

    smry = pd.Series(determine(g) for gid,g in labs.groupby('officer_name')) \
            .value_counts() \
            .reset_index(name='n', drop=False) \
            .pivot_table(columns='index', values='n') \
            .reset_index(drop=True)

    uparams = labs[params].dropna().drop_duplicates().reset_index(drop=True)
    assert len(uparams) == 1

    out = pd.concat([uparams, smry], axis=1)

    out['file'] = os.path.basename(args.input)
    out['filetype'] = doctype(args.input)
    out.to_parquet(args.output)


# done.
