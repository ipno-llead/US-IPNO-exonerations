import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from PIL import Image
import argparse
import logging
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()


def change_dir(df):
    df.loc[:, "label"] = df.label.astype(str).str.replace(r"1\.0", "", regex=True)
    df.loc[:, "img_filepath"] = df.png.str.replace(r"^(.+)", r"../../thumbnail_accordian/\1", regex=True)
    return df 

def generate_reports(df):
    dfa = df[(df.doc_type == "report")]
    
    dfb = df[~(df.doc_type == "report")]
    dfb["label"] = "0"
    dfb = dfb.sample(n=6500, random_state=42)

    df = pd.concat([dfa, dfb], axis=0)
    df.loc[:, "label"] = df.label.astype(int)
    return df 

def generate_transcripts(df):
    dfa = df[(df.doc_type == "transcript")]
    dfa["label"] = "1"
    print(dfa.shape)
    dfa = dfa.sample(n=2000, random_state=42)
    print(dfa.shape)
    
    dfb = df[~(df.doc_type == "transcript")]
    dfb["label"] = "0"
    dfb = dfb[(dfb.doc_type == "report")]
    print(dfb.shape)

    df = pd.concat([dfa, dfb], axis=0)
    df.loc[:, "label"] = df.label.astype(int)
    return df 

def generate_testimonies(df):    
    dfa = df[(df.doc_type == "testimony")]
    #~6000 pages
    dfa["label"] = "1"
    
    dfb = df[~(df.doc_type == "testimony")]
    dfb["label"] = "0"
    dfb = dfb.sample(n=6500, random_state=42)

    df = pd.concat([dfa, dfb], axis=0)
    df.loc[:, "label"] = df.label.astype(int)
    return df 

def model(df):
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

    pre_trained_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    pre_trained_model.trainable = False

    # Add classification head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = pre_trained_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model
    es_callback = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32,
                    epochs=10, callbacks=[es_callback])
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    X_train_preprocessed = preprocess_input(X_train)
    X_test_preprocessed = preprocess_input(X_test)
    train_features = base_model.predict(X_train_preprocessed)
    test_features = base_model.predict(X_test_preprocessed)

    # Flatten features
    train_features_flat = train_features.reshape(train_features.shape[0], -1)
    test_features_flat = test_features.reshape(test_features.shape[0], -1)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Combine extracted features with original features
    X_train_combined = np.hstack([X_train_flat, train_features_flat])
    X_test_combined = np.hstack([X_test_flat, test_features_flat])

    # Train random forest classifier with combined features
    rf_model_combined = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_combined.fit(X_train_combined, y_train)
    rf_y_pred_combined = rf_model_combined.predict(X_test_combined)
    rf_test_acc_combined = accuracy_score(y_test, rf_y_pred_combined)
    print('Random Forest with combined features test accuracy:', rf_test_acc_combined) 

    trained_model = rf_model_combined
    trained_model_accuracy = rf_test_acc_combined
    return trained_model, trained_model_accuracy


if __name__ == "__main__":
    df = pd.read_csv(args.input)
    df = change_dir(df)

    transcripts = generate_transcripts(df)
    trained_model, trained_model_accuracy = model(transcripts)
    logger.info("Random Forest Model's Accuracy: {:.4f}".format(trained_model_accuracy))

    logger.info("Saving best model to {}...".format(args.output))
    np.save(args.output, trained_model)
    logger.info("Done.")
