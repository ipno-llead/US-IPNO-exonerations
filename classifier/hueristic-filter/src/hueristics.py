# This script has been altered for testing
# the extract_reports func should include data from all func
# the non_testimonates func should be renamed to non_relevant 
# and it should ref strings in each relevant func

import pandas as pd
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--index")
parser.add_argument("--output")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_transcripts(df):
    df = df[df.filename.str.lower().str.contains(r"transcript") & df.filetype.str.contains("pdf")]
    df["doc_type"] = "transcript"
    return df 

def get_testimonies(df):
    df = df[df.filename.str.lower().str.contains(r"testimony") & df.filetype.str.contains("pdf")]
    df["doc_type"] = "testimony"
    df["label"] = "1"
    return df

def get_non_testimonies(df):
    df = df[~df.filename.str.lower().str.contains(r"testimony") & df.filetype.str.contains("pdf")]
    df["doc_type"] = "testimony"
    df["label"] = "0"
    return df

def get_police_reports(df):
    df = df[df.filename.str.lower().str.contains(r"(?:arrest report|"
                                                 r"nopd report|"
                                                 r"police report|"
                                                 r"supplemental report)") & df.filetype.str.contains("pdf")]
    df["doc_type"] = "report"
    return df 

def change_dir(df):
    df.loc[:, "filepath"] = df.filepath.str.replace(r"^(.+)", r"../../\1", regex=True)
    return df 

def extract_documents(index):
    logger.info("Extracting testimonies...")
    testimonies = df.loc[:].pipe(get_testimonies)
    logger.info("Extracting transcripts...")
    non_testimonies = df.loc[:].pipe(get_non_testimonies)
    return testimonies, non_testimonies

def change_dir(df):
    df.loc[:, "filepath"] = df.filepath.str.replace(r"^(.+)", r"../\1", regex=True)
    return df 

def concat_reports(*args):
    logger.info("Concatenating documents...")
    df = pd.concat([testimonies, non_testimonies])
    return df

if __name__ == "__main__":
    index = args.index
    df = pd.read_csv(index, sep="|")
    df = df.pipe(change_dir)
    output_path = args.output
    testimonies, non_testimonies = extract_documents(index)
    df = concat_reports(testimonies, non_testimonies)
    logger.info("Writing output file...")
    df.to_csv(output_path, index=False)
    logger.info("Done.")