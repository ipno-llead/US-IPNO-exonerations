import pandas as pd
import numpy as np
import argparse
import logging
import os
from fastai.vision.all import *
from fastai.vision.models import resnet34
from fastai.vision.learner import *
from fastai.vision.learner import cnn_learner
from fastai.vision.augment import Rotate, RandomCrop, Flip


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--labeled")
parser.add_argument("--nonlabeled")
parser.add_argument("--output")
args = parser.parse_args()


def change_dir(df):
    df.loc[:, "img_filepath"] = df.png.str.replace(r"^(.+)", r"../../thumbnail_accordian/\1", regex=True)
    df.loc[:, "img_filepath"] = df.img_filepath.fillna("").str.replace(r"\.\/nan", "", regex=True)
    df = df[~(df.img_filepath == "")]
    return df 


def assign_label_col(df):
    df.loc[:, "label"] = 0
    df.loc[:, "doc_type"] = "unknown"
    df.loc[:, "label"] = df.label.astype(int)
    return df 


def generate_reports(df):
    df = df[(df.doc_type == "report")]
    df.loc[:, "label"] = 1
    df.loc[:, "label"] = df.label.astype(int)
    return df 


def concat(dfa, dfb):
    report_uids = [x for x in dfa["filehash"]]
    logger.info(f"Number of labeled reports: {len(dfa)}")
                         
    dfb = dfb[~(dfb.filehash.isin(report_uids))]
    dfb = dfb.sample(n=1500, random_state=42)
    logger.info(f"Number of non-labeled samples: {len(dfb)}")

    df = pd.concat([dfa, dfb], axis=0)
    df = df[~(df.label.fillna("") == "")]
    df.loc[:, "label"] = df.label.astype(str).str.replace(r"(\.0)", "", regex=True).astype(int)
    logger.info(f"Number of samples in the final dataset: {len(df)}")
    logger.info(f"Class distribution:\n{df.label.value_counts(normalize=True)}")
    return df 


def get_dataloaders(data):
    train_aug = aug_transforms(mult=1.0, 
                               do_flip=True, 
                               flip_vert=False, 
                               max_rotate=90.0, 
                               min_zoom=1.0, 
                               max_zoom=1.1, 
                               max_lighting=0.2, 
                               max_warp=0.2, 
                               p_affine=0.75, 
                               p_lighting=0.75, 
                               xtra_tfms=[RandomCrop(size=(260, 260))])
    
    valid_aug = aug_transforms(mult=1.0, do_flip=False)

    dls = ImageDataLoaders.from_df(data,
            path="",
            fn_col='img_filepath',
            label_col='label',
            item_tfms=Resize((289, 221)),
            batch_tfms=[*aug_transforms(size=(221, 221), pad_mode='zeros'), Flip(p=0.5)],
            valid_pct=.4,
            batch_size=16,
            train_tfms=train_aug,
            valid_tfms=valid_aug
            )
    return dls


def setup_model(dls):
    gpu = torch.device("cuda")
    learn = vision_learner(dls,
            resnet34,
            metrics=[error_rate, accuracy],
            path=".",
            model_dir=".")
    learn.to(gpu)
    learn.dls.to(gpu)
    return learn

if __name__ == '__main__':
    tuning_n = 20

    df_lab = pd.read_csv(args.labeled)
    df_lab = change_dir(df_lab)
    reports = generate_reports(df_lab)
    
    df_nonlab = pd.read_csv(args.nonlabeled)
    df_nonlab = change_dir(df_nonlab)
    df_nonlab = assign_label_col(df_nonlab)
    
    train = concat(reports, df_nonlab)
    
    dls = get_dataloaders(train)
    learn = setup_model(dls)
    
    # Use lr_find to find optimal learning rate
    lr = learn.lr_find()
    logger.info(f"Optimal learning rate: {lr.valley}")
    
    # Train the model using the optimal learning rate
    learn.fine_tune(tuning_n, lr.valley)
    
    learn.export(args.output)
    
    # Output accuracy score based on validation data
    acc = learn.validate()[1]
    logger.info(f"Accuracy: {acc}")