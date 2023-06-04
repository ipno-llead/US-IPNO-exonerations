import pandas as pd
import torch
from torchvision.transforms.functional import resize
from fastai.vision.all import *
from fastai.vision.learner import load_learner
import argparse
import logging
import os
from PIL import Image

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
    df = df.sample(n=1000)
    return df


def apply_model(df, model):
    dls = ImageDataLoaders.from_df(df, path="", fn_col='img_filepath', bs=64, normalize=True,
                                   )
    dl = dls.test_dl(df)  # Create a test dataloader for inference

    # Use model.get_preds to get predictions for the test dataloader
    preds, _ = model.get_preds(dl=dl)

    # Iterate over the predictions and assign scores to the dataframe
    for i, image_path in enumerate(df["img_filepath"]):
        try:
            score = preds[i][1].item()  # Probability score for the positive class
            df.loc[df["img_filepath"] == image_path, "score"] = score
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")

    return df


if __name__ == '__main__':
    logger.info("Loading the model...")
    model = load_learner(args.model)

    logger.info("Setting up the device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    logger.info("Loading the index")
    index = pd.read_csv(args.index)
    index = change_dir(index)

    logger.info("Applying the model to each image...")
    try:
        index = apply_model(index, model)
    except Exception as e:
        logger.error(f"An error occurred while applying the model: {e}")

    logger.info("Saving the updated index...")
    try:
        index.to_csv(args.output, index=False)
    except Exception as e:
        logger.error(f"An error occurred while saving the updated index: {e}")

    logger.info("Done")