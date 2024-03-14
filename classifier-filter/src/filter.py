import argparse
import logging
import pandas as pd

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_hueristic_df(file_path):
    return pd.read_csv(file_path)

def read_classifier_df(file_path):
    return pd.read_csv(file_path)

def filter_hueristic_df(df):
    logging.info("Filtering heuristic dataframe based on doc_type='report'")
    df = df[df['doc_type'] == 'report']
    return df

def filtered_classifier_df(df):
    logging.info("Filtering classifier dataframe based on score >= 0.5")
    df = df[df['score'] >= 0.5]
    df = df[["filepath", "filename", "filehash", "filesize", "uid", "filetype", "case_id"]]
    df = df.drop_duplicates(["uid"])
    return df

def compare_dfs(hueristic_df, classifier_df):
    logging.info("Comparing heuristic and classifier dataframes to find unique UIDs in heuristic dataframe")
    uids_hueristic = set(hueristic_df['uid'])
    uids_classifier = set(classifier_df['uid'])
    unique_uids_hueristic = uids_hueristic - uids_classifier
    unique_uids_hueristic = list(unique_uids_hueristic)
    hueristic_df = hueristic_df[hueristic_df['uid'].isin(unique_uids_hueristic)]
    return hueristic_df

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description='Process and combine dataframes.')
    parser.add_argument('--hueristic_df', required=True, help='File path to the heuristic dataframe CSV')
    parser.add_argument('--classifier_df', required=True, help='File path to the classifier dataframe CSV')
    parser.add_argument('--output', required=True, help='Output file path for the combined CSV')

    args = parser.parse_args()

    logging.info("Reading heuristic dataframe from {}".format(args.hueristic_df))
    hueristic_df = read_hueristic_df(args.hueristic_df)
    hueristic_df = filter_hueristic_df(hueristic_df)

    logging.info("Reading classifier dataframe from {}".format(args.classifier_df))
    classifier_df = read_classifier_df(args.classifier_df)
    classifier_df = filtered_classifier_df(classifier_df)

    logging.info("Comparing dataframes to identify unique heuristic records")
    hueristic_df = compare_dfs(hueristic_df, classifier_df)

    logging.info("Combining dataframes")
    df = pd.concat([hueristic_df, classifier_df])

    logging.info("Writing combined dataframe to {}".format(args.output))
    df.to_csv(args.output, index=False)
