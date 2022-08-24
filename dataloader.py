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
    Format of return: (df, 
                       one-hot labels, 
                       text label: (ordinal label, onehot label),
                       ordinal label: (ordinal label, onehot label), 
                       ordinal label: (text label, onehot label)
                       )
    """
    if data_method not in ["minmax", "unnormalized", "zscore"]:
        print("Data method should be in [minmax, unnormalized, zscore]")
        return
    df_dir = os.path.join(base_dir, f"preprocessed_entire_dataset_{data_method}.csv")
    if not os.path.exists(df_dir):
        print(f"Cannot find dataset in {df_dir}")
        return

    df = pd.read_csv(df_dir)
    text_ordinal_mapping = {}
    for text, ordinal in zip(pd.unique(df['text_label']), pd.unique(df["ordinal_label"])):
        text_ordinal_mapping[text] = ordinal
    return tuple([df] + list(get_one_hot_labels_and_mappings(df["ordinal_label"], text_ordinal_mapping)))


def get_one_hot_labels_and_mappings(labels: pd.DataFrame, text_ordinal_mapping: dict) -> tuple:
    """
    Get one-hot encoded labels for dataset,
    along with mapping {ordinal_label: one-hot vector}

    :param labels: the label column of the DF to be encoded
    :param text_ordinal_mapping: {"text label": ordinal label} mapping
    :return: (
                one-hot labels,
                text_mapping: {text_label: (position of 1 in the one-hot vector, corresponding one-hot vector)},
                ordinal_mapping {ordinal_label: (position of 1 in the one-hot vector, corresponding one-hot vector)},
                one_pos_text_mapping {position of 1 in the one-hot vector, (text_label, corresponding one-hot vector)}
            ) tuple
    """
    encoder = OneHotEncoder(sparse=False)

    labels = np.asarray(labels).reshape(len(labels), 1)
    res = encoder.fit_transform(labels)

    ordinal_text_mapping = {v: k for k, v in text_ordinal_mapping.items()}

    ordinal_mapping = {}
    text_mapping = {}
    one_pos_text_mapping = {}
    one_hot_labels = np.unique(res, axis=0)
    ordinal_labels = encoder.inverse_transform(one_hot_labels).reshape(-1)
    for ordinal, one_hot in zip(ordinal_labels, one_hot_labels):
        pos_in_one_hot = np.where(one_hot == 1)[0][0]
        text = ordinal_text_mapping[ordinal]
        ordinal_mapping[ordinal] = pos_in_one_hot, one_hot
        text_mapping[text] = pos_in_one_hot, one_hot
        one_pos_text_mapping[pos_in_one_hot] = text, one_hot
    return res, text_mapping, ordinal_mapping, one_pos_text_mapping


def get_train_test_indices_for_all_folds(dataframe: pd.DataFrame, k: int = 3,
                                         seed: int = 2022,
                                         shuffle: bool = True):
    """
    Return the all train and test indices for all folders in cross validation.

    :param dataframe: the dataframe to be split
    :param k: the number of folders to
    :param seed: the random seed
    :param shuffle: whether shuffle each class’s samples before splitting into batches
    :return: the generator of train, test indices of all folders
    """
    k_fold = StratifiedKFold(n_splits=k, random_state=seed, shuffle=shuffle)
    return k_fold.split(X=dataframe, y=dataframe['ordinal_label'])


if __name__ == "__main__":
    # test functionality of methods listed in this script

    # zscore dataset, compute one-hot encodings and mapping
    df, one_hot_labels, text_mapping, ordinal_mapping, one_pos_text_mapping = get_data_frame(data_method='zscore')

    res = get_train_test_indices_for_all_folds(df)

    for train_indices, test_indices in res:
        print(train_indices.shape, test_indices.shape)
