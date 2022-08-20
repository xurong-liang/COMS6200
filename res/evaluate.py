import sklearn.metrics


def evaluate(classifier: str, hyper: str, y_true, y_preds, start: int, end: int):
    """
    Output results to a file

    
    :param (String): Name of classifier
    :param hyper (String): Name of parameter
    :param y_preds (np.array/torch tensor): model predictions
    :param y_true (np.array/torch tensor): ground truth labels
    :param start (int): start of training time
    :param end (int): end of training time 
    """

    acc = sklearn.metrics.accuracy_score(y_true, y_preds)
    precision = sklearn.metrics.precision_score(y_true, y_preds)
    recall = sklearn.metrics.recall_score(y_true, y_preds)
    f1 = sklearn.metrics.f1_score(y_true, y_preds)


    with open(classifier + hyper + '.txt', 'w') as f:
        print('Classifier: ', classifier, file=f)
        print('Hyperparameters: ', hyper, file=f)
        print('Precision: ', precision, file=f)
        print('Recall: ', recall, file=f)
        print('F1: ', f1, file=f)
        print('Accuracy: ', acc, file=f)
        print('Training time: ', end - start, "secs", file=f)