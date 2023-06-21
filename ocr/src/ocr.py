import re
import logging
import os
import hashlib
from functools import partial
from multiprocessing import Pool
from PyPDF2 import PdfFileReader
import pandas as pd
import pytesseract
from tqdm import tqdm
from pdf2image import convert_from_path
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--index")
parser.add_argument("--txtdir", default="output/txt")
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--output")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def pdf_length(filepath):
    try:
        with open(f'{filepath}', 'rb') as f:
            pdf = PdfFileReader(f)
            return pdf.getNumPages()
    except Exception as e:
        logging.error(f"Error processing PDF: {filepath} - {str(e)}")
        return None


def ocr_page(pageno, filename, engine, DPI, txtdir):
    txt_dir = os.path.join(args.txtdir, os.path.relpath(os.path.dirname(filename)), os.path.splitext(os.path.basename(filename))[0])
    os.makedirs(txt_dir, exist_ok=True)
    txt_filepath = os.path.join(txt_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_{pageno:04d}.txt')

    try:
        if os.path.exists(txt_filepath):  # Check if the text file already exists
            logging.info(f"Skipped page {pageno} of file: {filename} - Text file already exists in the path")
            return txt_filepath

        pages = convert_from_path(filename, dpi=DPI)
        img = pages[pageno-1]
        img = img.convert('RGB')

        with open(txt_filepath, 'w') as f:
            f.write(pytesseract.image_to_string(img))

        logging.info(f"Processed page {pageno} of file: {filename} - Text file saved at: {txt_filepath}")
        return txt_filepath
    except Exception as e:
        logging.error(f"Error during OCR for page {pageno} of file: {filename}\n{str(e)}")
        return None
    

def process_pdf(pdf_file):
    npages = pdf_length(pdf_file)
    if npages is None:
        logging.error(f"Skipped processing problematic PDF: {pdf_file}")
        return None
    logging.info(f"Processing {pdf_file} with {npages} pages")
    fileid = os.path.splitext(os.path.basename(pdf_file))[0]
    label = index.loc[index['filepath'] == pdf_file, 'label'].iloc[0]
    doc_type = index.loc[index['filepath'] == pdf_file, 'doc_type'].iloc[0]
    txt_filepaths = []
    for pageno in range(1, npages+1):
        txt_filepath = ocr_page(pageno, pdf_file, pytesseract, args.dpi, args.txtdir)
        if txt_filepath:
            txt_filepaths.append(txt_filepath)
    return pd.DataFrame({'filepath': pdf_file,
                         'fileid': fileid, 
                         'filename': os.path.basename(pdf_file),
                         'filesize': os.path.getsize(pdf_file),
                         'uid': index.loc[index['filepath'] == pdf_file, 'uid'].iloc[0],
                         'filetype': index.loc[index['filepath'] == pdf_file, 'filetype'].iloc[0],
                         'case_id': index.loc[index['filepath'] == pdf_file, 'case_id'].iloc[0],
                         'label': label,
                         'doc_type': doc_type,
                         'pageno': range(1, npages+1),
                         'txt_filepath': txt_filepaths})


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

        index = index[index['filetype']=='pdf']
        index = index[index['doc_type']=='transcript']
        logging.info(f"Number of PDF files in index: {len(index)}")

        pdf_files = index['filepath'].unique()
        logging.info(f"Number of unique PDF files in index: {len(pdf_files)}")

        dfs = []
        with Pool(processes=5) as p:
            for df in tqdm(p.imap(process_pdf, pdf_files), total=len(pdf_files)):
                if df is not None:
                    dfs.append(df)

        df = pd.concat(dfs, sort=False)

        df.to_csv(output_path, index=False)
        logging.info(f"CSV output saved to {output_path}")

        logging.info('done')
    except Exception as e:
        logging.error(f"An error occurred during the main execution:\n{str(e)}")
