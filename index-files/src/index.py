from os import listdir
from os.path import isfile, join
import hashlib 
import os
import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--outdir")
    return parser.parse_args()


def sha1(fname):
    hash_sha1 = hashlib.sha1()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            hash_sha1.update(chunk)
    return hash_sha1.hexdigest()


def generate_df(args):
    filehashes = []
    filepaths = []
    filenames = []
    filesizes = []

    for path, subdirs, files in os.walk(args.path):
        for name in files:
            filepath = os.path.join(path, name)
            fhash = sha1(filepath)
            
            file_size = os.path.getsize(filepath)
            filesizes.append(file_size)
            
            filehashes.append(fhash)
            filepaths.append(filepath)
            filenames.append(name)

    df = pd.DataFrame({"filepath": filepaths, "filename": filenames, "filehash": filehashes, "filesize": filesizes})
    return df 


def clean_df(df):
    df.loc[:, "filepath"] = df.filepath.str.replace(r"^\.\.\/\.\.\/", "", regex=True)
    
    uids = df.filehash.str.extract("^(\w{8})")
    df.loc[:, "uid"] = uids[0]
    
    filetypes = df.filename.str.extract("\.(\w{3,5})$")
    df.loc[:, "filetype"] = filetypes[0].str.lower()

    caseid = df.filepath.str.extract("wrongful-convictions-docs\/(\w+)\/")
    df.loc[:, "case_id"] = caseid[0]
    return df 


def check_duplicate_filehash(df):
    df = df.drop_duplicates(subset=["filehash"])
    for _ in df["filehash"]:
        unique = df.filehash.is_unique
        if unique is True:
            continue
        else:
            raise ValueError("duplicate filehash found")
    return df


def check_duplicate_uid(df):
    df = df.drop_duplicates(subset=["uid"])
    for _ in df["uid"]:
        unique = df.uid.is_unique
        if unique is True:
            continue
        else:
            raise ValueError("duplicate uid found")
    return df


if __name__ == "__main__":
    args = get_args()
    df = generate_df(args)
    df = clean_df(df)
    df = check_duplicate_filehash(df)
    df = check_duplicate_uid(df)
    df.to_csv(args.outdir, index=False, sep="|")
