from pdf2image import convert_from_path
import os
from os import path
import fitz
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
    
    
def pdf2png(pdf_path,dpi,png_path): 
    pages = convert_from_path(pdf_path,dpi)
    doc = fitz.open(pdf_path)
    page_number = 0
    return_list = []
    for page in pages:
        pdf_page = doc[page_number]
        page_number += 1
        str_num = str(page_number).zfill(4)
        page_directory = png_path+"_p"+str_num+".png"
        os.makedirs(os.path.dirname(page_directory), exist_ok=True)
        page.save(page_directory, 'PNG')
        return_list.append(page_directory)
    return return_list


def convert_pdf(row):
    logger.info("start convert pdf")
    row_list = row.to_list()
    first_entry = row_list[1]
    logger.info("found filepath column")
    full_path = first_entry[:2] + "/index-files" + first_entry[2:]
    uid = str(row_list[4])
    dir1 = "/" + uid[:2]
    dir2 = "/" + uid[2:4]
#    input_path = get_dir(first_entry)
    output_path = "output"+ dir1 + dir2 + "/" + uid
    return pdf2png(full_path,50,output_path)
    

if __name__ == "__main__":
    csv_path = args.input
    df = pd.read_csv(csv_path)

    logger.info("Creating png's...")
    df["png"] = df.apply(convert_pdf, axis=1)
    logger.info("finished creating png's...")
    df = df.explode("png")
    

    logger.info("Writing output file...")
    output_path = args.output
    df.to_csv(output_path, index=False)
    logger.info("Done.")