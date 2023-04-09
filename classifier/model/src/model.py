import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import argparse
import logging
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

df = pd.read_csv(args.input)

def change_dir(df):
    df.loc[:, "label"] = df.label.astype(str)
    df.loc[:, "txt_filepath"] = df.txt_filepath.str.replace(r"^(.+)", r"../ocr/\1", regex=True)
    df.loc[:, "img_filepath"] = df.img_filepath.str.replace(r"^(.+)", r"../ocr/\1", regex=True)
    return df 

df = change_dir(df)

num_classes = 10

X = []
y = []

for index, row in df.iterrows():
    if not os.path.exists(row['img_filepath']):
        logger.warning("Skipping file: {} does not exist".format(row['img_filepath']))
        continue
    img = Image.open(row['img_filepath']).resize((224,224))
    X.append(np.array(img))
    y.append(row['label'])

le = LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1,1)
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

# SVC
svc_model = SVC(kernel='rbf', gamma='scale', C=1.0)
logger.info("Training svc model...")
svc_model.fit(X_train.reshape(X_train.shape[0], -1), y_train.argmax(axis=1))
svc_y_pred = svc_model.predict(X_test.reshape(X_test.shape[0], -1))
svc_test_acc = accuracy_score(y_test.argmax(axis=1), svc_y_pred)
logger.info("svc test accuracy: {:.4f}".format(svc_test_acc))

# RForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
logger.info("Training rforest model...")
rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train.argmax(axis=1))
rf_y_pred = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
rf_test_acc = accuracy_score(y_test.argmax(axis=1), rf_y_pred)
logger.info("rforest test accuracy: {:.4f}".format(rf_test_acc))

# LRegression
lr_model = LogisticRegression(random_state=42)
logger.info("Training lregression model..")
lr_model.fit(X_train.reshape(X_train.shape[0], -1), y_train.argmax(axis=1))
lr_y_pred = lr_model.predict(X_test.reshape(X_test.shape[0], -1))
lr_test_acc = accuracy_score(y_test.argmax(axis=1), lr_y_pred)
logger.info("lregression test accuracy: {:.4f}".format(lr_test_acc))

models = {'svc': {'model': svc_model, 'accuracy': svc_test_acc},
          'rforests': {'model': rf_model, 'accuracy': rf_test_acc},
          'lregression': {'model': lr_model, 'accuracy': lr_test_acc}}

best_model_name = max(models, key=lambda x: models[x]['accuracy'])
best_model = models[best_model_name]['model']
best_accuracy = models[best_model_name]['accuracy']

logger.info("Best model: {}, Accuracy: {:.4f}".format(best_model_name, best_accuracy))

logger.info("Saving best model to {}...".format(args.output))
np.save(args.output, best_model)

logger.info("Done.")


