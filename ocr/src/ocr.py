import re
import logging
import os
import hashlib
from functools import partial
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from pdf2image import convert_from_path
import argparse
import json
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import time
import numpy as np 
import pdf2image
from PyPDF2 import PdfReader 

parser = argparse.ArgumentParser()
parser.add_argument("--index")
parser.add_argument("--jsondir", default="output/json")
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--output")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)



def pdf_length(filepath):
    try:
        with open(filepath, 'rb') as f:
            pdf = PdfReader(f)
            return len(pdf.pages)  # Use len(pdf.pages) to get the number of pages
    except Exception as e:
        logging.error(f"Error processing PDF: {filepath} - {str(e)}")
        return None

def getcreds():
    with open("src/credentials/creds_cv_new.txt", "r") as c:
        creds = c.readlines()
    return creds[0].strip(), creds[1].strip()

endpoint, key = getcreds()
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    
def process_pdf(pdf_file, index):
    npages = pdf_length(pdf_file)
    if npages is None:
        logging.error(f"Skipped processing problematic PDF: {pdf_file}")
        return None

    fileid = os.path.splitext(os.path.basename(pdf_file))[0]
    label = index.loc[index['filepath'] == pdf_file, 'label'].iloc[0]
    doc_type = index.loc[index['filepath'] == pdf_file, 'doc_type'].iloc[0]
    uid = index.loc[index['filepath'] == pdf_file, 'uid'].iloc[0]
    filetype = index.loc[index['filepath'] == pdf_file, 'filetype'].iloc[0]
    case_id = index.loc[index['filepath'] == pdf_file, 'case_id'].iloc[0]

    relative_pdf_path = os.path.relpath(pdf_file, start='../index-files/input')
    json_dir = os.path.join(args.jsondir, os.path.dirname(relative_pdf_path))
    os.makedirs(json_dir, exist_ok=True)
    json_filename = f"{fileid}.json"
    json_filepath = os.path.join(json_dir, json_filename)

    if os.path.exists(json_filepath):
        logging.info(f"Skipping {json_filepath}, file already exists")
        return pd.DataFrame([{
            'filepath': pdf_file,
            'fileid': fileid, 
            'filename': os.path.basename(pdf_file),
            'filesize': os.path.getsize(pdf_file),
            'uid': uid,
            'filetype': filetype,
            'case_id': case_id,
            'label': label,
            'doc_type': doc_type,
            'json_filepath': json_filepath
        }])

    all_messages = []
    with open(pdf_file, "rb") as file:
        pdf_data = file.read()
        for i in range(npages):
            try:
                image = convert_from_path(
                    pdf_data, dpi=args.dpi, first_page=i+1, last_page=i+1
                )[0]

                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                ocr_result = client.read_in_stream(img_byte_arr, raw=True)
                operation_id = ocr_result.headers["Operation-Location"].split("/")[-1]

                while True:
                    result = client.get_read_result(operation_id)
                    if result.status.lower() not in ["notstarted", "running"]:
                        break
                    time.sleep(1)

                if result.status.lower() == "failed":
                    logging.error(f"OCR failed for page {i+1} of file {pdf_file}")
                    continue

                page_content = "\n".join([" ".join([word.text for word in line.words]) for read_result in result.analyze_result.read_results for line in read_result.lines])
                all_messages.append({
                    "page_number": i + 1,
                    "page_content": page_content
                })

            except Exception as e:
                logging.error(f"Error processing page {i+1} of file {pdf_file}: {e}")

    with open(json_filepath, "w") as f:
        json.dump({"messages": all_messages}, f, indent=4)

    logging.info(f"Finished writing to {json_filepath}")
    
    return pd.DataFrame([{
        'filepath': pdf_file,
        'fileid': fileid, 
        'filename': os.path.basename(pdf_file),
        'filesize': os.path.getsize(pdf_file),
        'uid': uid,
        'filetype': filetype,
        'case_id': case_id,
        'label': label,
        'doc_type': doc_type,
        'json_filepath': json_filepath 
    }])



def change_fp(df):
    df.loc[:, "filepath"] = df.filepath.str.replace(r"^../(.+)", r"../index-files/\1", regex=True)
    return df 


if __name__ == '__main__':
    try:
        input_path = args.index
        output_path = args.output
        DPI = args.dpi

        index = pd.read_csv(input_path)
        index = index.pipe(change_fp)
        logging.info(f"Total number of files in index: {len(index)}")

        transcripts = index[index['filetype'] == 'pdf']

        logging.info(f"Number of PDF files in index after filtering: {len(index)}")

        pdf_files = index['filepath'].unique()
        logging.info(f"Number of unique PDF files in index: {len(pdf_files)}")

        dfs = []
        with Pool(processes=200) as p:
            for df in tqdm(p.imap(partial(process_pdf, index=index), pdf_files), total=len(pdf_files)):
                if df is not None:
                    dfs.append(df)

        if dfs:
            df = pd.concat(dfs, sort=False)
            df.to_csv(output_path, index=False)
            logging.info(f"CSV output saved to {output_path}")
        else:
            logging.warning("No PDF files were processed.")

        logging.info('done')
    except Exception as e:
        logging.error(f"An error occurred during the main execution:\n{str(e)}")

    client.close()