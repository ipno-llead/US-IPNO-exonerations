# vim: set ts=4 sts=0 sw=4 si fenc=utf-8 et:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:
# Authors:     TS
# Maintainers: TS
# Copyright:   2023, HRDAG, GPL v2 or later
# =========================================

import os
import argparse
from pathlib import Path


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir")
    parser.add_argument("--outputdir")
    return parser.parse_args()


def clean_name(fpath):
    return fpath \
            .replace(" ", "-") \
            .replace('(', '-') \
            .replace(')', '-') \
            .replace("'", '') \
            .lower()


def make_symlink(infile, outdir):
    """
    create a working symlink to `infile` in `outdir`
    the symlinks are `make`-friendly, meaning they have no spaces
    note: preserves the original dir structure of the infile, but replaces
          the root of the path to `infile` with `outdir`. for example, 
          `in/path/to/f name.ext` gets symlinked as `out/path/to/f-name.ext`

    - infile: file to symlink
    - outdir: root location to place output (the "output repo")
    """
    indir = Path(infile).parts[0]
    outfile = clean_name(infile.replace(f'{indir}/', f'{outdir}/'))
    output_dirname = os.path.dirname(outfile)
    os.makedirs(output_dirname, exist_ok=True)
    sl_target = os.path.relpath(infile, start=output_dirname)
    os.symlink(sl_target, outfile)


if __name__ == '__main__':
    args = getargs()

    counter = 0
    for path, dirs, files in os.walk(args.inputdir):
        for file in files:
            make_symlink(os.path.join(path, file), args.outputdir)
            counter += 1

    print(f'created {counter} symlinks')


# done.

