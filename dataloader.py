"""
This file is used to input dataset from csv file and generate Train/Test set for
sklearn and Pytorch
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_train_test_set(train_percentage: float = .8, path: str = "./dataset/",
                       force_build: bool = False, normalized: bool = True) -> tuple:
    """
    Return the training and testing dataset.

    The function will first find if there exists train.csv and test.csv files
    inside path. If no, it will then generate a training and testing dataset and
    will then save to path. If yes and force_build is False, it will read the train.csv
    and test.csv and return them. If force_build is enabled, sampling will always
    take effect and newly sampled training and testing dataset will be saved.

    :param normalized: whether the input file is normalized
    :param train_percentage: The percentage of records used for training
    :param path: The path to {train.csv, test.csv} or preprocessed_entire_dataset.csv
    :param force_build: whether we want resample train/test set
    :return: (training set, testing set), each of which is a pd.DataFrame
    """
    assert 0 < train_percentage < 1, "train percentage is invalid"
    train_path, test_path = os.path.join(path, "train.csv"), os.path.join(path, "test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("train.csv and test.csv found.")
        if not force_build:
            print("train.csv and test.csv loaded.")
            train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
            print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
            return train_df, test_df
        else:
            print("Resample train, test set...")

    if not force_build:
        print("Sampling train, test set...")

    print(f"Percentage of train dataset: {train_percentage}")

    if normalized:
        file_dir = os.path.join(path, "preprocessed_entire_dataset_normalized.csv")
        print("Input from normalized dataset")
    else:
        file_dir = os.path.join(path, "preprocessed_entire_dataset_unnormalized.csv")
        print("Input from not normalized dataset")

    dataset = load_csv(file_dir=file_dir)
    train_df, test_df = train_test_split(dataset,
                                         train_size=int(train_percentage * len(dataset)),
                                         random_state=2022, shuffle=True)
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    print(f"train, test set saved to {path}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df


def load_csv(file_dir: str = "./dataset/preprocessed_entire_dataset_normalized.csv") -> pd.DataFrame:
    """
    Load the processed DataFrame from csv file

    :param file_dir: location of the dataset to be loaded
    :return: The loaded DF
    """
    return pd.read_csv(file_dir)


def get_one_hot_labels_and_mapping(train_set: pd.DataFrame, test_set: pd.DataFrame) -> tuple:
    """
    Get one-hot encoded labels for training set and test_set,
    along with mapping {ordinal_label: one-hot vector}

    :param train_set: the training set
    :param test_set: the test set
    :return: (one-hot train labels, one-hot test labels, mapping {ordinal_label: one-hot vector})
    """
    train_size, test_size = len(train_set), len(test_set)
    encoder = OneHotEncoder(sparse=False)

    train_labels = np.asarray(train_set["ordinal_label"]).reshape(train_size, 1)
    encoder.fit(train_labels)
    one_hot_train_labels = encoder.transform(train_labels)

    encode_mapping = {}
    one_hot_labels = np.unique(one_hot_train_labels, axis=0)
    origin_labels = encoder.inverse_transform(one_hot_labels).reshape(-1)
    for origin, one_hot in zip(origin_labels, one_hot_labels):
        encode_mapping[origin] = one_hot

    test_labels = np.asarray(test_set["ordinal_label"]).reshape(test_size, 1)
    one_hot_test_labels = encoder.transform(test_labels)
    return one_hot_train_labels, one_hot_test_labels, encode_mapping


if __name__ == "__main__":
    # test get train test set
    train, test = get_train_test_set()
    # test get one-hot labels and ordinal label -> one-hot mapping
    one_hot_train, one_hot_test, one_hot_mapping = \
        get_one_hot_labels_and_mapping(train_set=train, test_set=test)
