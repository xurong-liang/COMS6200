"""
This file has methods related to model evaluation
"""

import os
import sklearn.metrics


def evaluate(classifier: str, hyper: str, y_true, y_preds,
             start: int, end: int, data_method: str, res_dir: str = "./res/"):
    """
    Evaluate the performance of a classifier, output the evaluation result to
    stdout and res_dir.

    The output file will have the name of {classifier}_{hyper}.txt

    :param classifier: Name of classifier
    :param hyper: Name of parameter
    :param y_preds: model predictions
    :param y_true: ground truth labels
    :param start: start of training time
    :param end: end of training time
    :param data_method: [unnormalized, minmax, zscore]
    :param res_dir: the directory where the result is saved
    """
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    acc = sklearn.metrics.accuracy_score(y_true, y_preds)
    precision = sklearn.metrics.precision_score(y_true, y_preds)
    recall = sklearn.metrics.recall_score(y_true, y_preds)
    f1 = sklearn.metrics.f1_score(y_true, y_preds)
    train_time = f"{end - start:.4f}s"

    text = ""
    for name in ['classifier', 'data_method', 'hyper', 'acc', 'precision', 'recall', 'f1', "train_time"]:
        metric = locals()[name]
        if type(metric) == str:
            text += f'{name.replace("_", " ").title()}: {metric}\n'
        else:
            text += f'{name.replace("_", " ").title()}: {metric:.4f}\n'

    print(text)

    file_path = os.path.join(res_dir, f"{classifier}_{hyper}_{data_method}.txt")
    with open(file_path, 'w') as f:
        print(text, file=f)


if __name__ == "__main__":
    # For testing only, not meant to be used in the project
    evaluate(classifier='svm', hyper="linear, lr_10", y_true=[1, 0, 0, 1], y_preds=[1, 0, 0, 0],
             start=60, end=90, data_method='zscore')
