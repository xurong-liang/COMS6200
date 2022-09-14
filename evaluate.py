"""
This file has methods related to model evaluation
"""

import os

import numpy as np
import sklearn.metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def compute_metric_values(y_true, y_preds):
    """
    Compute the performance metric values, given y_true and y_preds

    :param y_true: all ground truth labels
    :param y_preds: all predicted labels
    :return: dict with keys [accuracy, precision, recall and f1]
    """
    acc = sklearn.metrics.accuracy_score(y_true, y_preds)
    precision = sklearn.metrics.precision_score(y_true, y_preds)
    recall = sklearn.metrics.recall_score(y_true, y_preds)
    f1 = sklearn.metrics.f1_score(y_true, y_preds)
    return {
        "accuracy": acc,
        "precision": precision,
        'recall': recall,
        "f1": f1
    }


def save_result_text(classifier: str, hyper: str, data_method: str,
                     class_performance_text: str, res_dir: str = "./res/",
                     imbalanced_problem: bool = False):
    """
    Output the evaluation result to stdout and res_dir.
    The output file will have the name of {classifier}_{hyper}.txt

    :param class_performance_text:
    :param classifier: Name of classifier
    :param hyper: Name of parameter. If address imbalanced dataset,
     write method to address it as well: e.g. linear_undersampling
    :param data_method: [unnormalized, minmax, zscore]
    :param res_dir: the directory where the result is saved
    :param imbalanced_problem: if the result is computed to address the problem of
        imbalanced dataset, if yes, then save to res_dir/address_imbalanced_res/
    """
    if imbalanced_problem:
        res_dir = os.path.join(res_dir, "address_imbalanced_res")

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    text = ""
    for name in ['classifier', 'data_method', 'hyper']:
        metric = locals()[name]
        if type(metric) == str:
            text += f'{name.replace("_", " ").title()}: {metric}\n'
        else:
            text += f'{name.replace("_", " ").title()}: {metric:.4f}\n'

    text += class_performance_text
    print(text)

    file_path = os.path.join(res_dir, f"{classifier}_{hyper}_{data_method}.txt")
    with open(file_path, 'w') as f:
        print(text, file=f)


def generate_class_performance_text(res_dict: dict, imbalanced_problem: bool = False) -> str:
    """
    This method takes all classes in res_dict, and generate performance text
    for all the classes.

    :param res_dict: the result dict in form {"class_name": {"metric_name": "metric_val"}}
        e.g.
            {
                "normal": {
                            "accuracy": acc,
                            "precision": precision,
                            'recall': recall,
                            "f1": f1,
                            "training time": training time
                        }
            }
    :param imbalanced_problem: if the input results come from imbalanced problem computation
    :return: the performance text, each class in form
        class: xxx, [accuracy: xxx, precision: xxx, recall: xxx, f1: xxx, training time: xxxs]
    """
    text = ""
    if imbalanced_problem:
        entity_name = "method"
    else:
        entity_name = "class"

    for class_name, metrics in res_dict.items():
        text += f"{entity_name}: {class_name}, [\n"
        for count, item in enumerate(metrics.items()):
            metric_name, val = item
            if metric_name.lower().replace(" ", '_') == 'training_time':
                text += f"training time: {val:>.8f}s"
            elif metric_name.lower().replace(" ", '_') == "new_train_label_distributions":
                text += f"new train distribution: {val}"
            else:
                text += f"{metric_name}: {val:>.8f}"
            if count == len(metrics) - 1:
                text += "\n]\n"
            else:
                text += ",\n"
    return text


def plot_2_pc_results(dataset_x: dict, dataset_y: dict, res_dir: str = "./res/address_imbalanced_res"):
    """
    For each entry in the dataset dict, perform PCA with 2 principal components, and then save
    as image

    :param dataset_y: the dict that contains all y of original and balanced dataset, in form
                {
                       dataset balancing technique/origin: y from all batches combined
                                }
    :param dataset_x: the dict that contains all X of original and balanced dataset, in form
                {
                       dataset balancing technique/origin: X from all batches combined
                                }
    :param res_dir: the directory where the plot is saved
    """
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    for name, x in dataset_x.items():
        y = dataset_y[name]
        plt.figure()
        plt.title(f'{name} 2 PC')
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        new_x = PCA(n_components=2).fit_transform(x)
        for label in [1., 0.]:
            row_idx = np.where(y == label)[0]
            if name == "origin" and label == 1.:
                plt.scatter(new_x[row_idx, 0], new_x[row_idx, 1],
                            label=str(label), s=50, marker="X")
            else:
                plt.scatter(new_x[row_idx, 0], new_x[row_idx, 1], label=str(label))
        plt.legend()
        # in order to view image
        plt.savefig(os.path.join(res_dir, f"{name}.png"))
        # for usage in latex
        plt.savefig(os.path.join(res_dir, f"{name}.pgf"))


if __name__ == "__main__":
    # For testing only, not meant to be used in the project
    test_out = {
                "Normal": {
                            "accuracy": .555,
                            "precision": 1,
                            'recall': .997,
                            "f1": .777
                        },
                "DDos": {
                    "accuracy": .555,
                    "precision": 1,
                    'recall': .997,
                    "f1": .777
                }
            }

    performance_text = generate_class_performance_text(res_dict=test_out)
    save_result_text(classifier="SVM", hyper="linear, c=10", data_method="zscore",
                     class_performance_text=performance_text)
