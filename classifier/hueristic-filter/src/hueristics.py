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
    logger.info("Extracting transcripts...")
    transcripts = index.pipe(get_transcripts)
    logger.info("Extracting testimonies...")
    testimonies = index.pipe(get_testimonies)
    logger.info("Extracting police reports...")
    police_reports = index.pipe(get_police_reports)
    return transcripts, testimonies, police_reports

def change_dir(df):
    df.loc[:, "filepath"] = df.filepath.str.replace(r"^(.+)", r"../\1", regex=True)
    return df 

def concat_reports(*args):
    logger.info("Concatenating documents...")
    df = pd.concat([transcripts, testimonies, police_reports])
    return df

if __name__ == "__main__":
    index = args.index
    index = pd.read_csv(index, sep="|")
    index = index.pipe(change_dir)

    logger.info("Extracting relevant data...")
    transcripts, testimonies, police_reports = extract_documents(index)
    df = concat_reports(transcripts, testimonies, police_reports)

    logger.info("Writing output file...")
    output_path = args.output
    df.to_csv(output_path, index=False)
    logger.info("Done.")