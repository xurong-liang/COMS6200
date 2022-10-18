'''
This script conducts testing and evaluation of k-nn. Using the default of k=5
'''
from sklearn.neighbors import KNeighborsClassifier
from timeit import default_timer as timer
from dataloader import get_data_frame, get_train_test_indices_for_all_folds
from evaluate import compute_metric_values, generate_class_performance_text, save_result_text
import numpy as np
import copy
from collections import Counter
from multiprocessing import Process
from datetime import timedelta
from argparse import ArgumentParser
import sys
import imblearn
import os
import re
'''
def check_has_undersampling(classifier_name: str, imbalanced_class: str,
                            dataset: str,
                            res_dir: str = "./res/address_imbalanced_res") -> bool:
    """
    Check whether we have called undersampling for current (classifier, imbalanced calss)
    combination.
    Since for over-sampling we have multiple sampling strategies so solve_imbalance_problem()
    function will be called multiple times, but we do not have sampling strategies
    for under-sampling, meaning we only need to call once

    :param res_dir:
    :return:
    """
    if os.path.exists(res_dir):
        expr = f'{classifier_name}_{imbalanced_class}_(\w+)_AllKNN_(\w+)_'
        _, _, _files = next(os.walk(res_dir), (None, None, []))
        for _file in _files:
            if _file.endswith(f"{dataset}.txt") and re.match(expr, _file):
                return True
    return False
'''

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
    over_samplings = ["SMOTE", "SMOTETomek", "SMOTEENN", ] 
    under_samplings = ["AllKNN", "CondensedNearestNeighbour", "TomekLinks"]
    class_imbalanced_methods = over_samplings 

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
            model = KNeighborsClassifier()

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
    if set(class_imbalanced_methods) != set(under_samplings):
        # not purely under-samplings, add sampling strategy as hyper setting
        hyper_text += f"_sampling_strategy_{sampling_strategy}"
    print(performance_text, imbalanced_class)
    f = open("knn_imbalanced.txt", "a")
    f.write('sampling strategy: '+ sampling_strategy+ 'i_class: '+imbalanced_class+ "performance: "+ performance_text)
    f.close()
    '''
    save_result_text(classifier=base_classifier + f"_{imbalanced_class}",
                     hyper=hyper_text, data_method=dataset,
                     class_performance_text=performance_text,
                     res_dir= imbalanced_res_path)
    folder_name = base_classifier + f"_{imbalanced_class}_" + hyper_text + "_" + dataset
    # compute pca
    plot_2_pc_results(dataset_x=all_datasets_x, dataset_y=all_datasets_y,
                      res_dir=plot_path)
    '''

# def get_arguments() -> dict:
#     """
#     Initialize and get program input arguments
#     :return: set of argument values in a ditionary
#     """
#     parser = ArgumentParser()
#     parser.add_argument("--task", type=str,
#                         help="What is the task to be conducted."
#                              " Options: ['full', 'imbalance_problem']"
#                         )
#     parser.add_argument("--classifier", type=str, nargs="+",
#                         help="What classifier to be used. "
#                              "Options: ['adaboost', 'random_forest']"
#                         )
#     parser.add_argument("--dataset", type=str, nargs="+",
#                         help="The type of dataset to be evaluated. "
#                              "default: ['unnormalized', 'zscore', 'minmax']"
#                         )
#     parser.add_argument("--imbalanced_class", type=str,
#                         help="The name of the imbalanced class to be evaluated.")
#     parser.add_argument("--sampling_strategy", type=float, nargs="+",
#                         help="The list of sampling strategies for SMOTE to compute.")

#     task_range = ['full', 'imbalance_problem']
#     classifier_range = ['both', 'adaboost', 'random_forest']
#     dataset_range = ['unnormalized', 'zscore', 'minmax']
#     parser.set_defaults(
#         task="full",
#         classifier="both",
#         dataset=dataset_range,
#         imbalanced_class="U2R",
#         sampling_strategy=[.1 * k for k in range(1, 6)]
#     )
#     args = vars(parser.parse_args())
#     # capitalize first letter of each word
#     args["imbalanced_class"] = args["imbalanced_class"].title()

#     if args["task"] not in task_range:
#         print(f"task must be in {task_range}", file=sys.stderr)
#         exit(1)
#     for classifier in args['classifier']:
#         if classifier not in classifier_range:
#             print(f"classifier must be in {classifier_range}", file=sys.stderr)
#             exit(1)
#     for dataset in args["dataset"]:
#         if dataset not in dataset_range:
#             print(f"{dataset} is not a valid dataset", file=sys.stderr)
#             exit(1)
#     return args

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



def batch_runner(train_features, batch_train_labels, test_features,
                 batch_ground_truth, class_metrics, classifier_type,
                 seed, class_name) -> dict:
    """
    Runner of a batch

    :return: the updated class_metrics
    """
    if classifier_type == "knn":
        classifier= KNeighborsClassifier()
    '''
    if classifier_type == "svm-linear":
        classifier = svm.LinearSVC(random_state=seed)
    elif classifier_type == 'svm-rbf':
        classifier = svm.SVC(kernel='rbf', random_state=seed)
    elif classifier_type == 'svm-poly':
        classifier = svm.SVC(kernel='poly', random_state=seed)
    else:
        classifier = DecisionTreeClassifier(random_state=seed)
'''
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
    argparse = ArgumentParser()
    argparse.add_argument("--run_full_program", type=lambda x: x.lower() == "true", required=True,
    help="if true, then runs the whole program; otherwise, tries to address the imbalanced problem of U2R class")


    # True if run full program;¨False otherwise (i.e., imbalanced dataset problem)
    run_full_program = argparse.parse_args().run_full_program
    seed= 2022
    total_start = timer()
    #processes = []
    if run_full_program: 
        for method in ["minmax", "unnormalized", "zscore"]:
            for c_type in ['knn']:
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
                evaluate_a_data_frame(inputs)
    else:
        for _s in [round(.1 * _, 1) for _ in range(5, 6)]:
                print(f"current sampling strategy: {_s}")
                solve_imbalance_problem(dataset= 'minmax', base_classifier='knn', imbalanced_class= 'U2R',
                            sampling_strategy=_s)
           #p = Process(target=evaluate_a_data_frame, args=(inputs,))
            #print(f"Now start {c_type} on {method} dataset")
            #p.start()
            #processes.append(p)

    #for p in processes:
     #   p.join()
    total_end = timer()
    print(f'All processes finished, total elapsed time {timedelta(seconds=total_end - total_start)}.')
