"""
This script conducts testing and evaluation of ensemble methods, namely,
Adaboost and RandomForest, with base model of Decision Tree
"""
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from timeit import default_timer as timer
from dataloader import get_data_frame, get_train_test_indices_for_all_folds
from evaluate import compute_metric_values, generate_class_performance_text, save_result_text
import numpy as np
import copy
from multiprocessing import Process
from datetime import timedelta


def init_class_metrics(text_mapping: dict) -> dict:
    """
    Initialize the metrics dict that records all performance results

    :param text_mapping: {text_label: (position of 1 in the one-hot vector, corresponding one-hot vector)}
    :return: the result dict in form {"class_name": {"metric_name": []}}
    """
    metrics = {
        'accuracy': [],
        "precision": [],
        "recall": [],
        "f1": [],
        'training time': []
    }

    class_metrics = {}
    for class_name in text_mapping.keys():
        class_metrics[class_name] = copy.deepcopy(metrics)
    return class_metrics


def evaluate_a_data_frame(info: dict):
    """
    Process target function

    :param info: The dictionary that contains all information to evaluate a data frame
    """
    all_folder_indices = info["train_test_indices"]

    data_frame = info["data_frame"]
    data_method = info["data_method"]
    one_hot_labels = info["one_hot_labels"]
    one_pos_text_mapping = info['one_pos_text_mapping']
    classifier_type = info["classifier_type"]
    seed = info["seed"]
    text_mapping = info["text_mapping"]
    class_metrics = init_class_metrics(text_mapping)

    for train_idxes, test_idxes in all_folder_indices:
        train_features, train_labels = data_frame.iloc[train_idxes], one_hot_labels[train_idxes, :]
        test_features, ground_truth = data_frame.iloc[test_idxes], one_hot_labels[test_idxes, :]

        # one classifier per class
        for one_pos, item in one_pos_text_mapping.items():
            class_name = item[0]
            batch_train_labels = train_labels[:, one_pos]
            batch_ground_truth = ground_truth[:, one_pos]

            assert not np.all(batch_ground_truth == 0) and not np.all(batch_train_labels == 0)
            if class_name == "Normal":
                # compute all combined scores
                # normal class gets label 0, other classes all 1's
                batch_train_labels = np.bitwise_not(batch_train_labels.astype(bool)).astype(float)
                batch_ground_truth = np.bitwise_not(batch_ground_truth.astype(bool)).astype(float)
            class_metrics = batch_runner(train_features, batch_train_labels,
                                         test_features, batch_ground_truth, class_metrics,
                                         classifier_type, seed, class_name=class_name)

    for text_label in text_mapping.keys():
        metrics = class_metrics[text_label]
        for key, value in metrics.items():
            metrics[key] = np.array(value)
            if key == "training time":
                metrics[key] = metrics[key].sum()
            else:
                metrics[key] = metrics[key].mean()

    performance_text = generate_class_performance_text(res_dict=class_metrics)
    save_result_text(classifier=classifier_type, hyper="default", data_method=data_method,
                     class_performance_text=performance_text)
    print(f'{classifier_type} results for {data_method} dataset completed.')


def batch_runner(train_features, batch_train_labels, test_features,
                 batch_ground_truth, class_metrics, classifier_type,
                 seed, class_name) -> dict:
    """
    Runner of a batch

    :return: the updated class_metrics
    """
    if classifier_type == "random_forest":
        classifier = RandomForestClassifier(random_state=seed)
    else:
        classifier = AdaBoostClassifier(random_state=seed)

    elapsed_time, classifier = train_a_batch(features=train_features,
                                             labels=batch_train_labels,
                                             model=classifier)
    class_metrics[class_name]['training time'].append(elapsed_time)

    class_result = test_a_batch(features=test_features,
                                ground_truth=batch_ground_truth,
                                model=classifier)
    for metric, value in class_result.items():
        class_metrics[class_name][metric].append(value)
    return class_metrics


def train_a_batch(features, labels, model):
    """
    The function that fit a train batch to the model, and then return it

    :param features: the set of training features for current batch
    :param labels: the set of labels
    :param model: the model to fit
    :return: training time in seconds, trained model
    """
    assert len(features) == len(labels), "Length of features and labels mismatched."
    start_time = timer()
    model.fit(X=features, y=labels)
    end_time = timer()
    return end_time - start_time, model


def test_a_batch(features, ground_truth, model):
    """
    Past the batch of test features to model and compute metric evaluations
    with respect to ground_truth.

    :param features: the set of test features
    :param ground_truth: the set of ground truth values
    :param model: the model to be tested
    :return: evaluation results
    """
    assert len(features) == len(ground_truth), "Length of features and ground truth mismatched."
    scores = model.predict(features)
    results = compute_metric_values(y_true=ground_truth, y_preds=scores)
    return results


if __name__ == "__main__":
    total_start = timer()
    processes = []

    for method in ["minmax", "unnormalized", "zscore"]:
        for c_type in ['random_forest', "adaboost"]:
            # params = df, one_hot_labels, text_mapping, ordinal_mapping, one_pos_text_mapping
            params = get_data_frame(data_method=method)
            indices_loader = get_train_test_indices_for_all_folds(params[0])

            inputs = {
                "train_test_indices": indices_loader,
                "data_frame": params[0].drop(columns=["text_label", "ordinal_label"]),
                "one_hot_labels": params[1],
                'text_mapping': params[2],
                'one_pos_text_mapping': params[4],
                "classifier_type": c_type,
                "seed": 2022,
                "data_method": method
            }
            p = Process(target=evaluate_a_data_frame, args=(inputs,))
            print(f"Now start {c_type} on {method} dataset")
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
    total_end = timer()
    print(f'All processes finished, total elapsed time {timedelta(seconds=total_end - total_start)}.')
