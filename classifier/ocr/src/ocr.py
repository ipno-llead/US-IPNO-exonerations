import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--index")
parser.add_argument("--txtdir", default="output/txt300")
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--output")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def pdf_length(filepath):
    with open(f'{filepath}', 'rb') as f:
        pdf = PdfFileReader(f)
        return pdf.getNumPages()

def process_pdf(pdf_file):
    pdf_filehash = index.loc[index['filepath'] == pdf_file, 'filehash'].iloc[0]
    npages = pdf_length(pdf_file)
    logging.info(f"Processing {pdf_file} with {npages} pages")
    return pd.DataFrame({'fileid': os.path.splitext(os.path.basename(pdf_file))[0], 
                         'filehash': pdf_filehash,
                         'pageno': range(1, npages+1),
                         'text': list(map(partial(ocr_cached, filename=pdf_file, engine=pytesseract, DPI=DPI, txtdir=args.txtdir), 
                                             range(1, npages+1)))})

def ocr_cached(pageno, filename, engine, DPI, txtdir):
    txt_fn = os.path.join(txtdir, f'{os.path.splitext(os.path.basename(filename))[0]}_{pageno:04d}.txt')
    if os.path.exists(txt_fn):
        with open(txt_fn, 'r') as f:
            return f.read()
    logging.info(f'OCR for: {txt_fn}')
    pages = convert_from_path(filename, dpi=DPI)
    img = pages[pageno-1]
    img = img.convert('RGB')
    img.save(os.path.join(txtdir, f'{os.path.splitext(os.path.basename(filename))[0]}_{pageno:04d}.jpg'), 'JPEG')
    txt = engine.image_to_string(img)
    with open(txt_fn, 'w') as f:
        f.write(txt)
    return txt

def change_fp(df):
    df.loc[:, "filepath"] = df.filepath.str.replace(r"^(.+)", r"../../\1", regex=True)
    return df 

if __name__ == '__main__':
    input_path = args.index
    output_path = args.output
    DPI = args.dpi

    index = pd.read_csv(input_path)
    index = index.pipe(change_fp)
    logging.info(f"Total number of files in index: {len(index)}")

    index = index[index['filetype']=='pdf']
    logging.info(f"Number of PDF files in index: {len(index)}")

    pdf_files = index['filepath'].unique()
    logging.info(f"Number of unique PDF files in index: {len(pdf_files)}")

    dfs = []
    with Pool() as p:
        for df in tqdm(p.imap(process_pdf, pdf_files), total=len(pdf_files)):
            dfs.append(df)

    df = pd.concat(dfs, sort=False)
    
    df.to_csv(output_path, index=False)
    logging.info(f"CSV output saved to {output_path}")

    logging.info('done')
