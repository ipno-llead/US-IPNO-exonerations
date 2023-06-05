import pandas as pd
import torch
from torchvision.transforms.functional import resize
from fastai.vision.all import *
from fastai.vision.learner import load_learner
import argparse
import logging
import os
from PIL import Image
from PIL import UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None


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

from PIL import UnidentifiedImageError

def apply_trained_model(df, model_path, device):
    model = load_learner(model_path, cpu=False)
    dls = ImageDataLoaders.from_df(df, path="", fn_col='img_filepath', bs=64, item_tfms=Resize((224, 224)))
    dl = dls.test_dl(df)
    dl = dl.to(device)  # Move the data to the correct device
    preds, _ = model.get_preds(dl=dl)

    for i, image_path in enumerate(df["img_filepath"]):
        try:
            score = preds[i][1].item()
            df.loc[df["img_filepath"] == image_path, "score"] = score
        except UnidentifiedImageError as e:
            logger.warning(f"Unidentified image error: {e}")
            df.loc[df["img_filepath"] == image_path, "score"] = None
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")
            df.loc[df["img_filepath"] == image_path, "score"] = None

    return df


if __name__ == '__main__':
    logger.info("Loading the model...")
    model_path = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading the index")
    index = pd.read_csv(args.index)
    index = change_dir(index)

    logger.info("Applying the model to each image...")
    try:
        index = apply_trained_model(index, model_path, device)  # Pass the device as an argument
    except Exception as e:
        logger.error(f"An error occurred while applying the model: {e}")

    logger.info("Saving the updated index...")
    try:
        output_path = args.output
        index.to_csv(output_path, index=False)
    except Exception as e:
        logger.error(f"An error occurred while saving the updated index: {e}")

    logger.info("Done")
