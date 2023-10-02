"""
preprocess kdd 2012 track 2 dataset, and create train/test split

Prerequisite, download and unzip track2.zip from
https://www.kaggle.com/competitions/kddcup2012-track2/overview
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def main():
    names = [
        "click",
        "impression",
        "display_url",
        "ad_id",
        "advertiser_id",
        "depth",
        "position",
        "query_id",
        "keyword_id",
        "title_id",
        "description_id",
        "user_id"
    ]

    df = pd.read_csv(
        "track2/training.txt",
        sep="\t",
        header=None,
        names=names,
        # work with sub-sample of the raw data
        nrows=1000000
    )
    df["label"] = np.where(df["click"] > 0, 1, 0)
    print(df.shape)
    print(df.head())

    df_title, df_query = preprocess_df_entity_token_id()
    df_user = pd.read_csv(
        "track2/userid_profile.txt",
        sep="\t",
        header=None,
        names=["user_id", "gender", "age"]
    )

    df_enriched = (
        df
            .merge(df_title, on="title_id")
            .merge(df_query, on="query_id")
            .merge(df_user, on="user_id", how="left")
    )
    df_final = preprocess_tabular_features(df_enriched)
    print(df_final.shape)
    print(df_final.head())
    print(df_final["label"].value_counts())
    df_train, df_test = train_test_split(df_final, test_size=0.1, random_state=1234, stratify=df["label"])

    df_train.to_parquet("track2_processed_train.parquet", index=False)
    df_test.to_parquet("track2_processed_test.parquet", index=False)


def preprocess_tabular_features(df: pd.DataFrame):
    """Ordinal encoder categorical features as well as scale numerical features"""
    numerical_features = ["depth"]
    categorical_features = ["gender", "age", "advertiser_id", "user_id"]

    df[numerical_features] = df[numerical_features].fillna(0)
    df[categorical_features] = df[categorical_features].fillna(-1)

    ordinal_encoder = OrdinalEncoder(min_frequency=30)
    df[categorical_features] = ordinal_encoder.fit_transform(df[categorical_features])

    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    df[numerical_features] = min_max_scaler.fit_transform(df[numerical_features])
    return df


def preprocess_df_entity_token_id():
    """create vocabulary id from raw token id"""
    df_title = pd.read_csv(
        "track2/titleid_tokensid.txt",
        sep="\t",
        header=None,
        names=["entity_id", "tokens_id"]
    )
    df_title["entity"] = "title"

    df_query = pd.read_csv(
        "track2/queryid_tokensid.txt",
        sep="\t",
        header=None,
        names=["entity_id", "tokens_id"]
    )
    df_query["entity"] = "query"

    df_entity_token_id = pd.concat([df_title, df_query], axis=0).reset_index(drop=True)
    print(df_entity_token_id.shape)
    print(df_entity_token_id.head())

    # use tf-idf to create the token to vocabulary mapping
    # The default regexp selects tokens of 2 or more alphanumeric characters
    # (punctuation is completely ignored and always treated as a token separator),
    # here we modified it to 1, else single digit tokens would get skipped,
    # we'll need to provide additional filtering arguments like min and max df, else
    # the word level vocabulary would become extremely big
    token_pattern = r"(?u)\b\w+\b"
    tfidf_vect = TfidfVectorizer(token_pattern=token_pattern, min_df=10, max_df=0.5)
    tfidf_vect.fit(df_entity_token_id["tokens_id"])
    # original vocab size 1049677
    vocab_size = len(tfidf_vect.vocabulary_)
    print("vocab size: ", vocab_size)

    vocab_id = []
    for token_id in df_entity_token_id["tokens_id"]:
        vocab = [tfidf_vect.vocabulary_.get(token, vocab_size) for token in token_id.split("|")]
        vocab_id.append(vocab)

    df_entity_token_id["vocab_id"] = vocab_id

    df_title = df_entity_token_id.loc[df_entity_token_id["entity"] == "title", ["entity_id", "vocab_id"]]
    df_title = df_title.rename(columns={"entity_id": "title_id", "vocab_id": "tokenized_title"})
    df_query = df_entity_token_id.loc[df_entity_token_id["entity"] == "query", ["entity_id", "vocab_id"]]
    df_query = df_query.rename(columns={"entity_id": "query_id", "vocab_id": "tokenized_query"})
    return df_title, df_query


if __name__ == "__main__":
    main()