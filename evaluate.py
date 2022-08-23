"""
This file has methods related to model evaluation
"""

import os
import sklearn.metrics


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
                     class_performance_text: str, res_dir: str = "./res/"):
    """
    Output the evaluation result to stdout and res_dir.
    The output file will have the name of {classifier}_{hyper}.txt

    :param class_performance_text:
    :param classifier: Name of classifier
    :param hyper: Name of parameter
    :param data_method: [unnormalized, minmax, zscore]
    :param res_dir: the directory where the result is saved
    """
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


def generate_class_performance_text(res_dict: dict) -> str:
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
    :return: the performance text, each class in form
        class: xxx, [accuracy: xxx, precision: xxx, recall: xxx, f1: xxx, training time: xxxs]
    """
    text = ""
    for class_name, metrics in res_dict.items():
        text += f"class: {class_name}, ["
        for count, item in enumerate(metrics.items()):
            metric_name, val = item
            if metric_name.lower().replace(" ", '_') == 'training_time':
                text += f"training time: {val:>.4f}s"
            else:
                text += f"{metric_name}: {val:>.4f}"
            if count == len(metrics) - 1:
                text += "]\n"
            else:
                text += ", "
    return text


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
