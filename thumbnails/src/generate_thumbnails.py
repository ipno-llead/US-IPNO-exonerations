import os
import pandas as pd
import logging
import argparse
import fitz
import multiprocessing as mp

from pdf2image import convert_from_path, exceptions

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
parser.add_argument("--num_processes", type=int, default=56)
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def pdf2png(doc, dpi, png_path, image_size):
    if os.path.exists(png_path):
        logger.info(f"{png_path} already exists, skipping...")
        return [png_path]

    return_list = []
    for page_number, page in enumerate(doc):
        str_num = str(page_number + 1).zfill(4)
        page_directory = png_path + "_p" + str_num + ".png"

        if os.path.exists(page_directory):
            logger.info(f"{page_directory} already exists, skipping...")
            return_list.append(page_directory)
            continue

        os.makedirs(os.path.dirname(page_directory), exist_ok=True)
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            with open(page_directory, "wb") as f:
                pix.save(f, "png")
            return_list.append(page_directory)
        except Exception as e:
            logger.warning(f"Skipping page {page_number} of PDF {doc.name} because of exception: {e}")

    return return_list


def convert_pdf(row):
    row_list = row.to_list()
    first_entry = row_list[0]

    full_path = first_entry[:2] + "/index-files" + first_entry[2:]
    uid = str(row_list[4])
    dir1 = "/" + uid[:2]
    dir2 = "/" + uid[2:4]
    output_path = "output" + dir1 + dir2 + "/" + uid
    print(output_path)

    try:
        with fitz.open(full_path) as doc:
            logger.info(f"Converting PDF {full_path} to PNG...")
            # Adjust the DPI and image size values as per your requirements
            dpi = 300
            image_size = (224, 224)
            return pdf2png(doc, dpi, output_path, image_size)
    except Exception as e:
        logger.warning(f"Skipping PDF {full_path} because of exception: {e}")
        return []


def change_dir(df):
    df.loc[:, "filepath"] = df.filepath.str.replace(r"^(.+)", r"../\1", regex=True)
    return df


def filter_for_pdfs(df):
    df = df[df.filetype == "pdf"]
    df_splice = df[10000:]
    print(df_splice.shape)
    print(df_splice.head(10))
    return df


if __name__ == "__main__":
    csv_path = args.input
    df = pd.read_csv(csv_path, sep="|")
    df = df.pipe(change_dir).pipe(filter_for_pdfs)

    logger.info("Creating png's...")

    with mp.Pool(processes=args.num_processes) as pool:
        df["png"] = pool.map(convert_pdf, [row for idx, row in df.iterrows()])

    logger.info("Finished creating png's...")
    df = df.explode("png")

    logger.info("Writing output file...")
    output_path = args.output
    df.to_csv(output_path, index=False)
    logger.info("Done.")
