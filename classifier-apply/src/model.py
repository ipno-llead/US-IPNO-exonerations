import pandas as pd
import torch
from fastai.vision.all import *
import argparse
import logging
import concurrent.futures

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--index")
parser.add_argument("--output")
args = parser.parse_args()

def change_dir(df):
    df.loc[:, "img_filepath"] = df.png.str.replace(r"^(.+)", r"../thumbnails/\1", regex=True)
    df.loc[:, "img_filepath"] = df.img_filepath.fillna("").str.replace(r"\.\/nan", "", regex=True)
    df = df[~(df.img_filepath == "")]
    return df

def apply_model(df, model_path, max_workers=50):
    # Load the trained model
    learn = load_learner(model_path, cpu=False)

    def process_row(row):
        img_filepath = row['img_filepath']
        try:
            score = learn.predict(img_filepath)[2][1].item()
        except Exception as e:
            logger.warning(f"Error processing image {img_filepath}: {e}")
            score = 'NA'
        return score

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]

        scores = [future.result() for future in concurrent.futures.as_completed(futures)]

    df['score'] = scores

    return df

if __name__ == "__main__":
    # Load the model
    logger.info("Loading the model...")
    model_path = args.model

    # Load the index
    logger.info("Loading the index...")
    index = pd.read_csv(args.index)
    index = change_dir(index)

    # Apply the model to each image
    logger.info("Applying the model to each image...")
    try:
        index = apply_model(index, model_path)
    except Exception as e:
        logger.error(f"An error occurred while applying the model: {e}")

    # Save the updated index
    logger.info("Saving the updated index...")
    try:
        output_path = args.output
        index.to_csv(output_path, index=False)
    except Exception as e:
        logger.error(f"An error occurred while saving the updated index: {e}")

    logger.info("Done")
