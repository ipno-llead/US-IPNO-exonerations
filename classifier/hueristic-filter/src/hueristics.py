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
    return df 

def get_police_reports(df):
    # how to deal with duplicates in Robert Jones' case? 
    df = df[df.filename.str.lower().str.contains(r"(?:arrest report|"
                                                 r"nopd report|"
                                                 r"police report|"
                                                 r"supplemental report)") & df.filetype.str.contains("pdf")]
    df["doc_type"] = "report"
    return df 

def change_dir(df):
    df.loc[:, "filepath"] = df.filepath.str.replace(r"^(.+)", r"../../\1", regex=True)
    return df 

def extract_reports(index):
    logger.info("Loading index file...")
    df = pd.read_csv(index, on_bad_lines="skip", sep="|")
    logger.info("Extracting transcripts...")
    transcripts = df.loc[:].pipe(get_transcripts)
    logger.info("Extracting testimonies...")
    testimonies = df.loc[:].pipe(get_testimonies)
    logger.info("Extracting police reports...")
    police_reports = df.loc[:].pipe(get_police_reports)
    return transcripts, testimonies, police_reports

def concat_reports(*args):
    logger.info("Concatenating documents...")
    df = pd.concat([transcripts, testimonies, police_reports])
    return df 

if __name__ == "__main__":
    index = args.index
    output_path = args.output
    transcripts, testimonies, police_reports = extract_reports(index)
    df = concat_reports(transcripts, testimonies, police_reports)
    logger.info("Writing output file...")
    df.to_csv(output_path, index=False)
    logger.info("Done.")
