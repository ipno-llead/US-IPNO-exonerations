import pandas as pd

def read_hueristic_df():
    df = pd.read_csv("../output/hueristic-df.csv")
    return df 

def filter_hueristic_df(df):
    hueristic_df = hueristic_df[hueristic_df['doc_type'] == 'report']
    return hueristic_df


def read_classifier_df():
    df = pd.read_csv("../output/index_reports.csv")
    return df 

def filtered_classifier_df(df):
    classifier_df = classifier_df[classifier_df['score'] >= 0.5]

    classifier_df = classifier_df[["filepath", "filename", "filehash", "filesize", "uid", "filetype", "case_id"]]

    classifier_df = classifier_df.drop_duplicates(["uid"])
    return classifier_df


def compare_dfs(hueristic_df, classifier_df):
    uids_hueristic = set(hueristic_df['uid'])
    uids_classifer = set(classifier_df['uid'])

    unique_uids_hueristic = uids_hueristic - uids_classifer

    unique_uids_hueristic = list(unique_uids_hueristic) 

    hueristic_df = hueristic_df[hueristic_df['uid'].isin(unique_uids_hueristic)]    
    return hueristic_df


if __name__ == "__main__": 
    hueristic_df = read_hueristic_df()
    hueristic_df = filter_hueristic_df(hueristic_df)

    classifier_df = read_classifier_df()
    classifier_df = filtered_classifier_df(classifier_df)

    hueristic_df = compare_dfs(hueristic_df, classifier_df)

    df = pd.concat([hueristic_df, classifier_df])