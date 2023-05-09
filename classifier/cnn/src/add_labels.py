import os
from os import path
import pandas as pd

import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

#def get_dir(path):
#    f_path = "/home/jargentino/projects/US-IPNO-exonerations/index-files/" + path
#    return f_path
    
    

def add_label(row):
    if row["doc_type"] == "transcript":
        return(2)
    elif row["doc_type"] == "testimony":
        return(1)
    else:
        return(0)
        

if __name__ == "__main__":
    csv_path = args.input
    df = pd.read_csv(csv_path)

    logger.info("Adding labels")
    df["label"] = df.apply(add_label, axis=1)
    logger.info("finished adding labels...")
    logger.info("Writing output file...")
    output_path = args.output
    df.to_csv(output_path, index=False)
    logger.info("Done.")