import pandas as pd
import torch
from fastai.vision.all import *
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--index")
parser.add_argument("--output")
args = parser.parse_args()

def change_dir(df):
    df.loc[:, "img_filepath"] = df.png.str.replace(r"^(.+)", r"../../thumbnail_accordian/\1", regex=True)
    df.loc[:, "img_filepath"] = df.img_filepath.fillna("").str.replace(r"\.\/nan", "", regex=True)
    df = df[~(df.img_filepath == "")]
    return df

def apply_model(df, model_path):
    # Load the trained model
    learn = load_learner(model_path, cpu=False)

    # Predict the class for each image in the dataframe
    for idx, row in df.iterrows():
        try:
            df.loc[idx, 'score'] = learn.predict(row['img_filepath'])[2][1].item()
        except Exception as e:
            df.loc[idx, 'score'] = 'NA'
            logger.warning(f"Error processing image {row['img_filepath']}: {e}")

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
