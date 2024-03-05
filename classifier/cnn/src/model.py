import pandas as pd
import numpy as np
import argparse
import logging
import os
from fastai.vision.all import *
from fastai.vision.models import resnet34
from fastai.vision.learner import *
from fastai.vision.learner import cnn_learner
from fastai.vision.augment import *
from fastai.metrics import Precision, Recall, F1Score
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--labeled")
parser.add_argument("--nonlabeled")
parser.add_argument("--output")
args = parser.parse_args()


def change_dir(df):
    df.loc[:, "img_filepath"] = df.png.str.replace(r"^(.+)", r"../../thumbnails/\1", regex=True)
    df.loc[:, "img_filepath"] = df.img_filepath.fillna("").str.replace(r"\.\/nan", "", regex=True)
    df = df[~(df.img_filepath == "")]
    return df


def assign_label_col(df):
    print("Original DataFrame:")
    print(df)
    
    # Drop rows where 'filepath' contains 'testimony' or 'report', and explicitly create a copy
    filter_condition = ~(df['filepath'].str.contains('transcript|testimony', case=False))
    df_filtered = df[filter_condition].copy()
    
    # Assign initial values
    df_filtered.loc[:, "label"] = 0
    df_filtered.loc[:, "doc_type"] = "unknown"
    df_filtered.loc[:, "label"] = df_filtered.label.astype(int)
    
    print("Filtered DataFrame:")
    print(df_filtered)
    
    return df_filtered



def generate_labeled_data(df):
    df = df[(df.doc_type == "transcript")]
    df.loc[:, "label"] = 1
    df.loc[:, "label"] = df.label.astype(int)
    return df


def concat(dfa, dfb):
    report_uids = [x for x in dfa["filehash"]]
    logger.info(f"Number of labeled reports: {len(dfa)}")

    dfb = dfb[~(dfb.filehash.isin(report_uids))]
    dfb = dfb.sample(n=30000)  # Sample a smaller number of non-labeled samples
    logger.info(f"Number of non-labeled samples: {len(dfb)}")

    df = pd.concat([dfa, dfb], axis=0)
    df = df[~(df.label.fillna("") == "")]
    df.loc[:, "label"] = df.label.astype(str).str.replace(r"(\.0)", "", regex=True).astype(int)
    logger.info(f"Number of samples in the final dataset: {len(df)}")
    logger.info(f"Class distribution:\n{df.label.value_counts(normalize=True)}")
    return df


def get_dataloaders(data):
    train_tfms = aug_transforms(
        do_flip=True,
        flip_vert=True,
        max_rotate=15.0,
        max_zoom=1.1,
        max_lighting=0.2,
        max_warp=0.2,
        p_affine=0.75,
        p_lighting=0.75,
    )
    valid_tfms = []

    dls = ImageDataLoaders.from_df(
        data,
        path="",
        fn_col="img_filepath",
        label_col="label",
        item_tfms=Resize((224, 224)),  # Resize the images to (224, 224)
        batch_size=16,
        train_tfms=train_tfms,
        valid_tfms=valid_tfms,
    )
    return dls


def setup_model(dls):
    gpu = torch.device("cuda")
    learn = cnn_learner(
        dls,
        resnet34,
        metrics=[error_rate, accuracy, Precision(), Recall(), F1Score()],
        path=".",
        model_dir=".",
        pretrained=True,
        ps=0.5,  # Dropout probability
    )
    learn.to(gpu)
    learn.dls.device = gpu
    return learn


def evaluate_model(learn):
    values = learn.validate()
    loss = values[0]
    metrics = values[1:]
    metric_names = [m.name for m in learn.metrics]
    logger.info(f"Validation loss: {loss}")
    for metric_name, metric_value in zip(metric_names, metrics):
        logger.info(f"{metric_name.capitalize()}: {metric_value}")
    return loss, metrics


def train_model(learn):
    learn.fine_tune(5)
    return learn


if __name__ == "__main__":
    # Load and preprocess labeled data
    df_lab = pd.read_csv(args.labeled)
    df_lab = change_dir(df_lab)
    reports = generate_labeled_data(df_lab)

    # Load and preprocess non-labeled data
    df_nonlab = pd.read_csv(args.nonlabeled)
    df_nonlab = change_dir(df_nonlab)
    df_nonlab = assign_label_col(df_nonlab)

    # Concatenate the datasets
    train = concat(reports, df_nonlab)

    # Get data loaders
    dls = get_dataloaders(train)

    # Setup the model
    learn = setup_model(dls)

    # Train the model
    learn = train_model(learn)

    # Evaluate the model
    loss, metrics = evaluate_model(learn)

    # Save the trained model
    learn.export(args.output)
