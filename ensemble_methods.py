"""
This script conducts testing and evaluation of ensemble methods, namely,
Adaboost and RandomForest, with base model of Decision Tree
"""
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from collections import Counter
from timeit import default_timer as timer
from dataloader import get_data_frame, get_train_test_indices_for_all_folds
from evaluate import compute_metric_values, generate_class_performance_text,\
    save_result_text, plot_2_pc_results
import numpy as np
import copy
from multiprocessing import Process
from datetime import timedelta
from argparse import ArgumentParser
import sys
import imblearn


def solve_imbalance_problem(dataset: str, base_classifier: str, imbalanced_class: str,
                            sampling_strategy: float = .2):
    """
    Use under-/over-sampling strategy to

    :param dataset: the dataset to be evaluated.
    :param base_classifier: the classifier to be used.
    :param imbalanced_class: the imbalanced class to be evaluated.
    :param sampling_strategy: the percentage of minor class instances wrt to number
        of instances in the majority class
    """
    class_imbalanced_methods = ["SMOTE", "SMOTETomek", "SMOTEENN"]
    class_imbalanced_methods = ["AllKNN", "CondensedNearestNeighbour", "TomekLinks"]
    class_imbalanced_methods_mapping = {
        # over-sampling strategies
        "SMOTE": imblearn.over_sampling,
        "SMOTETomek": imblearn.combine,
        "SMOTEENN": imblearn.combine,
        # under-sampling strategies
        "AllKNN": imblearn.under_sampling,
        "CondensedNearestNeighbour": imblearn.under_sampling,
        "TomekLinks": imblearn.under_sampling
    }

    if base_classifier == "adaboost":
        raise NotImplementedError
    # here we implement RandomForest only since it is the only ensemble method
    # that yields performance of 47% on U2R
    class_metrics = init_class_metrics({k: None for k in class_imbalanced_methods},
                                       imbalanced_problem=True)
    origin_data_frame, one_hot_labels, text_label_mapping, _, _ = get_data_frame(data_method=dataset)

    data_frame = origin_data_frame.drop(columns=["text_label", "ordinal_label"])
    class_one_hot_pos, class_one_hot_encoding = text_label_mapping[imbalanced_class]

    # record all dataset for PCA computation
    all_datasets_x = {k: None for k in class_imbalanced_methods + ["origin"]}
    all_datasets_y = {k: None for k in class_imbalanced_methods + ["origin"]}
    indices_loader = get_train_test_indices_for_all_folds(origin_data_frame)
    for train_idxes, _ in indices_loader:
        train_features, train_labels = data_frame.iloc[train_idxes], one_hot_labels[train_idxes, :]
        batch_train_labels = train_labels[:, class_one_hot_pos]
        assert not np.all(batch_train_labels == 0)
        if all_datasets_x["origin"] is None:
            all_datasets_x['origin'] = np.array(train_features)
            all_datasets_y["origin"] = batch_train_labels
        else:
            all_datasets_x["origin"] = np.concatenate((all_datasets_x["origin"],
                                                      np.array(train_features)), axis=0)
            all_datasets_y["origin"] = np.concatenate((all_datasets_y["origin"],
                                                      batch_train_labels), axis=0)

    original_train_label_distributions = None
    for method in class_imbalanced_methods:
        original_train_label_distributions = []
        print(f'Running {method}...')
        indices_loader = get_train_test_indices_for_all_folds(origin_data_frame)
        for train_idxes, test_idxes in indices_loader:
            train_features, train_labels = data_frame.iloc[train_idxes], one_hot_labels[train_idxes, :]
            test_features, ground_truth = data_frame.iloc[test_idxes], one_hot_labels[test_idxes, :]
            batch_train_labels = train_labels[:, class_one_hot_pos]
            batch_ground_truth = ground_truth[:, class_one_hot_pos]
            assert not np.all(batch_ground_truth == 0) and not np.all(batch_train_labels == 0)

            original_train_label_distributions.append(dict(Counter(batch_train_labels)))
            imbalanced_start = timer()
            if method.startswith('S'):
                # oversampling
                X, y = getattr(class_imbalanced_methods_mapping[method], method) \
                    (sampling_strategy=sampling_strategy, random_state=seed). \
                    fit_resample(X=train_features, y=batch_train_labels)
            else:
                # under-sampling
                X, y = getattr(class_imbalanced_methods_mapping[method], method)(). \
                    fit_resample(X=train_features, y=batch_train_labels)
            assert not np.all(y == 0)

            if all_datasets_x[method] is None:
                all_datasets_x[method] = np.array(X)
                all_datasets_y[method] = y
            else:
                all_datasets_x[method] = np.concatenate((all_datasets_x[method],
                                                        np.array(X)), axis=0)
                all_datasets_y[method] = np.concatenate((all_datasets_y[method],
                                                        y), axis=0)

            class_metrics[method]["new_train_label_distributions"].append(dict(Counter(y)))
            imbalanced_end = timer()
            model = RandomForestClassifier(random_state=seed)

            # train model
            elapsed_time, model = train_a_batch(features=X,
                                                labels=y,
                                                model=model)

            # training elapsed time now include time taken on imbalanced dataset fixing
            elapsed_time += imbalanced_end - imbalanced_start
            class_metrics[method]['training time'].append(elapsed_time)
            # test model
            class_result = test_a_batch(features=test_features,
                                        ground_truth=batch_ground_truth,
                                        model=model)
            for metric, value in class_result.items():
                class_metrics[method][metric].append(value)

    for method in class_imbalanced_methods:
        metrics = class_metrics[method]
        for key, value in metrics.items():
            metrics[key] = np.array(value)
            if key == "training time":
                metrics[key] = metrics[key].sum()
            elif key == "new_train_label_distributions":
                continue
            else:
                metrics[key] = metrics[key].mean()

    performance_text = f"original train distribution: {original_train_label_distributions}\n\n"
    performance_text += generate_class_performance_text(res_dict=class_metrics,
                                                        imbalanced_problem=True)

    hyper_text = f"classifier_default_methods_" + "_".join(class_imbalanced_methods)
    hyper_text += f"_sampling_strategy_{sampling_strategy}"
    save_result_text(classifier=base_classifier + f"_{imbalanced_class}",
                     hyper=hyper_text, data_method=dataset,
                     class_performance_text=performance_text,
                     imbalanced_problem=True)
    folder_name = base_classifier + f"_{imbalanced_class}_" + hyper_text + "_" + dataset
    # compute pca
    plot_2_pc_results(dataset_x=all_datasets_x, dataset_y=all_datasets_y,
                      res_dir=f"./res/address_imbalanced_res/{folder_name}")


def get_arguments() -> dict:
    """
    Initialize and get program input arguments
    :return: set of argument values in a ditionary
    """
    parser = ArgumentParser()
    parser.add_argument("--task", type=str,
                        help="What is the task to be conducted."
                             " Options: ['full', 'imbalance_problem']"
                        )
    parser.add_argument("--classifier", type=str, nargs="+",
                        help="What classifier to be used. "
                             "Options: ['adaboost', 'random_forest']"
                        )
    parser.add_argument("--dataset", type=str, nargs="+",
                        help="The type of dataset to be evaluated. "
                             "default: ['unnormalized', 'zscore', 'minmax']"
                        )
    parser.add_argument("--imbalanced_class", type=str,
                        help="The name of the imbalanced class to be evaluated.")
    parser.add_argument("--sampling_strategy", type=float, nargs="+",
                        help="The list of sampling strategies for SMOTE to compute.")

    task_range = ['full', 'imbalance_problem']
    classifier_range = ['both', 'adaboost', 'random_forest']
    dataset_range = ['unnormalized', 'zscore', 'minmax']
    parser.set_defaults(
        task="full",
        classifier="both",
        dataset=dataset_range,
        imbalanced_class="U2R",
        sampling_strategy=[.1 * k for k in range(1, 6)]
    )
    args = vars(parser.parse_args())
    # capitalize first letter of each word
    args["imbalanced_class"] = args["imbalanced_class"].title()

    if args["task"] not in task_range:
        print(f"task must be in {task_range}", file=sys.stderr)
        exit(1)
    for classifier in args['classifier']:
        if classifier not in classifier_range:
            print(f"classifier must be in {classifier_range}", file=sys.stderr)
            exit(1)
    for dataset in args["dataset"]:
        if dataset not in dataset_range:
            print(f"{dataset} is not a valid dataset", file=sys.stderr)
            exit(1)
    return args


def init_class_metrics(text_mapping: dict, imbalanced_problem: bool = False) -> dict:
    """
    Initialize the metrics dict that records all performance results

    :param text_mapping: {text_label: (position of 1 in the one-hot vector, corresponding one-hot vector)}
    :param imbalanced_problem: if the class metrics is set for imbalanced problem
    :return: the result dict in form {"class_name": {"metric_name": []}}
    """
    metrics = {
        'accuracy': [],
        "precision": [],
        "recall": [],
        "f1": [],
        'training time': []
    }
    # records the number of class labels for each batch, after imbalanced method applied
    if imbalanced_problem:
        metrics["new_train_label_distributions"] = []

    class_metrics = {}
    for class_name in text_mapping.keys():
        class_metrics[class_name] = copy.deepcopy(metrics)
    return class_metrics


def run_full_program(datasets: list, classifiers: list):
    """
    The function that compute performance for all datasets and all ensemble classifiers

    :param datasets: the datasets to be evaluated
    :param classifiers: the classifiers to be evaluated
    """
    processes = []

    for method in datasets:
        for c_type in classifiers:
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
                "data_method": method
            }
            p = Process(target=evaluate_a_data_frame, args=(inputs,))
            print(f"Now start {c_type} on {method} dataset")
            p.start()
            processes.append(p)
    for p in processes:
        p.join()


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
                                         classifier_type, class_name=class_name)

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
                 class_name) -> dict:
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
    arguments = get_arguments()
    seed = 2022
    total_start = timer()
    if arguments["task"] == "full":
        # run both classifiers
        run_full_program(datasets=arguments["dataset"], classifiers=arguments["classifier"])
    elif arguments["task"] == "imbalance_problem":
        # address U2R's imbalanced dataset problem
        for _d in arguments["dataset"]:
            for _c in arguments["classifier"]:
                for strategy in arguments["sampling_strategy"]:
                    print(f"solve imbalance problem on {_d} dataset, {_c} classifier, "
                          f"sampling strategy {strategy}...")
                    solve_imbalance_problem(dataset=_d, base_classifier=_c,
                                            imbalanced_class=arguments["imbalanced_class"],
                                            sampling_strategy=strategy)
    total_end = timer()
    print(f'All processes finished, total elapsed time {timedelta(seconds=total_end - total_start)}.')
