"""
This file is used to input dataset from csv file and generate Train/Test set for
sklearn and Pytorch
"""
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold


def get_data_frame(base_dir: str = "./dataset/", data_method: str = 'minmax'):
    """
    Load the processed DataFrame from csv file

    :param base_dir: location of the dataset directory
    :param data_method: the dataset to be imported, options: [minmax, unnormalized, zscore]
    :return: (the loaded DF, one-hot-encoding arrays, mapping {ordinal_label: one-hot vector})
    """
    if data_method not in ["minmax", "unnormalized", "zscore"]:
        print("Data method should be in [minmax, unnormalized, zscore]")
        return
    df_dir = os.path.join(base_dir, f"preprocessed_entire_dataset_{data_method}.csv")
    if not os.path.exists(df_dir):
        print(f"Cannot find dataset in {df_dir}")
        return

    df = pd.read_csv(df_dir)

    one_hot_labels, one_hot_mapping = get_one_hot_labels_and_mapping(df["ordinal_label"])

    return df, one_hot_labels, one_hot_mapping


def get_one_hot_labels_and_mapping(labels: pd.DataFrame) -> tuple:
    """
    Get one-hot encoded labels for dataset,
    along with mapping {ordinal_label: one-hot vector}

    :param labels: the label column of the DF to be encoded
    :return: (one-hot labels, mapping {ordinal_label: one-hot vector})
    """
    encoder = OneHotEncoder(sparse=False)

    labels = np.asarray(labels).reshape(len(labels), 1)
    one_hot_labels = encoder.fit_transform(labels)

    encode_mapping = {}
    one_hot_labels = np.unique(one_hot_labels, axis=0)
    origin_labels = encoder.inverse_transform(one_hot_labels).reshape(-1)
    for origin, one_hot in zip(origin_labels, one_hot_labels):
        encode_mapping[origin] = one_hot
    return one_hot_labels, encode_mapping


def get_train_test_indices_for_all_folds(dataframe: pd.DataFrame, k: int = 3,
                                         seed: int = 2022,
                                         shuffle: bool = True):
    """
    Return the all train and test indices for all folders in cross validation.

    :param dataframe: the dataframe to be split
    :param k: the number of folders to
    :param seed: the random seed
    :param shuffle: whether shuffle each class’s samples before splitting into batches
    :return:
    """
    k_fold = StratifiedKFold(n_splits=k, random_state=seed, shuffle=shuffle)
    return k_fold.split(X=dataframe, y=dataframe['ordinal_label'])


if __name__ == "__main__":
    # test functionality of methods listed in this script

    # zscore dataset, compute one-hot encodings and mapping
    df, one_hot_labels, one_hot_mapping = get_data_frame(data_method='zscore')

    res = get_train_test_indices_for_all_folds(df)

    for train_indices, test_indices in res:
        print(train_indices.shape, test_indices.shape)
