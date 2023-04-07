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


# Command line args
parser = argparse.ArgumentParser()
parser.add_argument("--index")
parser.add_argument("--txtdir", default="output/txt300")
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--output")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

def pdf_length(filepath):
    with open(f'{filepath}', 'rb') as f:
        pdf = PdfFileReader(f)
        return pdf.getNumPages()

def process_pdf(row, txtdir, dpi):
    doc = os.path.basename(row['filepath'])
    expected_hash = row['filehash']
    npages = pdf_length(row['filepath'])
    return pd.DataFrame({'filehash': expected_hash, 
                         'pageno': range(1, npages+1),
                         'text': map(partial(ocr_cached, filename=row['filepath'], 
                                             engine=pytesseract, DPI=dpi, txtdir=txtdir), 
                                     range(1, npages+1))})


def ocr_cached(pageno, filename, engine, DPI, txtdir):
    txt_fn = os.path.join(txtdir, f'{pageno:04d}.txt')
    if os.path.exists(txt_fn):
        with open(txt_fn, 'r') as f:
            return f.read()
    logging.info(f'OCR for: {txt_fn}')
    pages = convert_from_path(filename, dpi=DPI)
    img = pages[pageno-1]
    img = img.convert('RGB')
    img.save(f'{pageno:04d}.jpg', 'JPEG')
    txt = engine.image_to_string(img)
    with open(txt_fn, 'w') as f:
        f.write(txt)
    return txt

if __name__ == '__main__':
    input_path = args.index
    output_path = args.output
    txtdir = args.txtdir
    DPI = args.dpi

    index = pd.read_csv(input_path)
    index = index[index['filetype']=='pdf']
    index['npages'] = index['filepath'].apply(pdf_length)

    index['text'] = pd.concat([df for _, df in index.groupby('filehash', sort=False)
                               .apply(lambda x: process_pdf(x.iloc[0], txtdir, DPI))
                               .groupby('filehash', sort=False)])

    out = index[['fileid', 'pageno', 'text']].explode('pageno').reset_index(drop=True)

    out.to_csv(output_path, index=False)

    logging.info('done')
