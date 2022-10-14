"""
Implement the Multilayer Perception (MLP) Classifier using Pytorch
Functions: loading data, training, testing and saving results
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataloader import *
from tqdm import tqdm
from evaluate import *
import time
import imblearn
from collections import Counter
from argparse import ArgumentParser
import sys


def reset_weights(m):
    """
    Reset model weights to avoid weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class MLP(nn.Module):
    """
    Multilayer perceptron model

    :param input_dim: number of features in the dataset
    :param output_dim: number of classes (1 for binary - probability between 0 and 1)
    """
    def __init__(self, input_dim=58, output_dim=1):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 80)
        self.hidden_fc = nn.Linear(80, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        """
        Forward the data over the model
        
        :param x: datapoints
        :return prediction probability between 0 and 1
        """
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = torch.sigmoid(self.output_fc(h_2))
        return torch.flatten(y_pred)

class MyDataset(Dataset):
    """
    Dataset class, to be used with DataLoader

    :param X: Datapoints
    :param y: labels
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class Runner():
    """
    Runner class, handle training and testing and saving results

    :param epoch: Number of epoch in training
    :param batch_size: Number of datapoints in a batch
    :param ordinal_label: Numerical label (between 0 and 7 in InSDN dataset)
    :param data_method: the dataset to be imported, options: [minmax, unnormalized, zscore] 
    :param imd_class: Class of under/oversampling method. Default: SMOTE (oversampling)
    :param imb_text: String representing the sampling method. Default: "SMOTE"
    """
    def __init__(self, epoch=10, 
                batch_size=128, 
                ordinal_label=0, 
                data_method="minmax", 
                imb_class=imblearn.over_sampling.SMOTE(sampling_strategy=0.1, 
                                    random_state=2022), 
                imb_text="SMOTE"):

        self.imb_class = imb_class
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.df, self.onehot, self.label2, self.label3, self.label4 = get_data_frame(data_method=data_method)
        self.X = self.df.drop(columns=["text_label", "ordinal_label"]).to_numpy()
        self.Y = self.onehot[:, ordinal_label]
        # original train distribution: [{0.0: 229248, 1.0: 11}, {0.0: 229248, 1.0: 11}, {0.0: 229248, 1.0: 12}]
        self.imb_text = imb_text
        self.orig_dist = []
        self.new_dist = []
        self.X_dict = {k: None for k in [imb_text] + ["origin"]}
        self.y_dict = {k: None for k in [imb_text] + ["origin"]}
        # del self.df
        

    def get_text_label(self): 
        """
        Get the text label of the current class for classification
        """
        return self.text_label


    def run(self):
        """
        Model training, with k-fold cross validation

        :return a dict with average metrics (acc, precision, recall, f1) of all k-folds 
        """
        folds = get_train_test_indices_for_all_folds(self.df, k=3)
        start = time.time()
        result_folds = []            

        for fold, (train_idx, test_idx) in enumerate(folds):
            X_train, y_train = self.X[train_idx, :], self.Y[train_idx]
            self.orig_dist.append(dict(Counter(y_train)))
            X_test, y_test = self.X[test_idx, :], self.Y[test_idx]

            if self.X_dict["origin"] is None:
                self.X_dict["origin"] = X_train
                self.y_dict["origin"] = y_train
            else:
                self.X_dict["origin"] = np.concatenate((self.X_dict["origin"], X_train), axis=0)
                self.y_dict["origin"] = np.concatenate((self.y_dict["origin"], y_train), axis=0)

            # resampling
            X_train_sample, y_train_sample = self.imb_class.fit_resample(X=X_train, y=y_train)
            self.new_dist.append(dict(Counter(y_train_sample)))
            if self.X_dict[self.imb_text] is None:
                self.X_dict[self.imb_text] = X_train_sample
                self.y_dict[self.imb_text] = y_train_sample
            else:
                self.X_dict[self.imb_text] = np.concatenate((self.X_dict[self.imb_text], X_train_sample), axis=0)
                self.y_dict[self.imb_text] = np.concatenate((self.y_dict[self.imb_text], y_train_sample), axis=0)

        
            # TODO: PCA plotting 

            del X_train, y_train
            
            train_ds = MyDataset(X_train_sample.astype(np.float32), y_train_sample.astype(np.float32))
            test_ds = MyDataset(X_test.astype(np.float32), y_test.astype(np.float32))


            train_loader = DataLoader(train_ds, 
                                      batch_size=self.batch_size)
            # faster to test on a single batch of every test data points
            test_loader = DataLoader(test_ds,
                                    batch_size=len(y_test))

            model = MLP()
            model.to(self.device)
            model.apply(reset_weights)
            optimizer = torch.optim.Adam(params=model.parameters())
            loss_fn=F.binary_cross_entropy

            for epoch in range(self.epoch):
                print(f'---epoch no---{epoch+1}---')
                self.train(train_loader, model, optimizer, loss_fn)

            metrics_dict = self.test(fold, test_loader, model, loss_fn)
            result_folds.append(metrics_dict)

        end = time.time()
        return self.save_results(result_folds, start, end)


    def save_results(self, result_folds: list, start: float, end: float):
        """
        Save average metrics (acc, precision, recall, f1) of all k-folds  

        :param results_folds: a list containing results of all folds
        :param start: start time of k-fold training
        :param end: end time of k-fold training
        
        :return a dict with average metrics (acc, precision, recall, f1) of all k-folds 
        """
        dict_metrics = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            dict_metrics[metric] = []

        for result in result_folds:
            for metric in ["accuracy", "precision", "recall", "f1"]:
                dict_metrics[metric].append(result[metric])

        for metric in ["accuracy", "precision", "recall", "f1"]:
            all_metrics = dict_metrics[metric]
            dict_metrics[metric] = np.asarray(all_metrics).mean()

        # dict_metrics["original_train_distribution"] = self.orig_dist
        dict_metrics["Training time"] = round(end - start, 2)
        dict_metrics["new_train_label_distributions"] = self.new_dist
        return dict_metrics


    def train(self, train_loader, model, optimizer, loss_fn):
        """
        Model training function

        :param train_loader: pytorch train data loader
        :param model: pytorch ML model
        :param optimizer: pytorch optimizer function
        :param loss_fn: pytorch loss function
        """
        model.train()
        for (data, target) in tqdm(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    def test(self, fold, test_loader, model, loss_fn):
        """
        Model testing function

        :param fold: number of fold
        :param test_loader: pytorch test data loader
        :param model: pytorch ML model
        :param loss_fn: pytorch loss function

        :return metric dictionary (acc rec prec f1)
        """
        model.eval()
        loss = 0
        correct = 0
        target = None
        pred = None
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss += loss_fn(output, target, reduction='sum').item()  
                pred = torch.where(output >= 0.5, 1, 0) 
                correct += pred.eq(target.view_as(pred)).sum().item()

        length = len(test_loader.dataset)
        loss /= length
        text = ""
        text += f'\nTest set for fold {fold+1}: Average loss: {loss:.4f}, '
        text += f'Accuracy: {correct}/{length} ({ correct / length:.8f})\n'
        print(text)

        if self.device.type == 'cuda':
            target, pred = target.cpu(), pred.cpu()
        metrics_dict = compute_metric_values(target.detach().numpy(), 
                                             pred.detach().numpy())
        return metrics_dict

def get_arguments() -> dict:
    """
    Initialize and get program input arguments
    :return: set of argument values in a dictionary
    """
    parser = ArgumentParser()
    parser.add_argument("--norm", type=str, nargs="+",
                        help="The type of normalization to be evaluated. "
                             "default: ['minmax']"
                        )

    parser.add_argument("--epoch", type=int,
                        help="Number of epoch in MLP training. "
                             "default: 5"
                        )

    parser.add_argument("--imbalanced_class", type=str,
                        help="The name of the imbalanced class to be evaluated."
                             "Default: U2R")

    parser.add_argument("--sampling_strategy", type=float, nargs="+",
                        help="The list of sampling strategies for SMOTE to compute."
                             "Default: 0.1, 0.2, 0.3, 0.4, 0.5")

    norms = ['minmax']
    parser.set_defaults(
        norm=norms,
        imbalanced_class="U2R",
        sampling_strategy=[.1 * k for k in range(1, 6)],
        epoch=5,
    )
    args = vars(parser.parse_args())
    # capitalize first letter of each word
    args["imbalanced_class"] = args["imbalanced_class"].title()

    
    if type(args["epoch"]) != int:
        print("Epoch must be an integer", file=sys.stderr)
        exit(1)

    for norm in args["norm"]:
        if norm not in norms:
            print(f"{norm} is not a valid normalization method", file=sys.stderr)
            exit(1)
    return args  


def main():
    """
    Main function of MLP training model in InSDN project
    """
    class_dict = {
                'BOTNET': 7,
                'Web-Attack': 6,
                'Probe': 5,
                'DoS': 4,
                'DDoS': 3,
                'BFA': 2,
                'U2R': 1,
                'Normal': 0,
    }
    
    args = get_arguments()
    print(args)

    # Only use minmax for testing sampling methods
    norms = args['norm']
    epoch = args['epoch']
    sampling_strategies = args['sampling_strategy']
    # Only use U2R attack class for testing sampling methods
    label = class_dict[args['imbalanced_class']]

    class_metrics = {}
    seed = 2022
    
    class_oversampling_methods = ["SMOTE", "SMOTETomek", "SMOTEENN"]
    class_undersampling_methods = ["AllKNN", "CondensedNearestNeighbour", "TomekLinks"]
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
    
    start = time.time()
    for norm in norms:
        print("Undersampling methods\n----------------\n")
        for imb_method in class_undersampling_methods:
            print(f"---Testing with \"{imb_method}\" imbalance method---")
            
            imb_class= getattr(class_imbalanced_methods_mapping[imb_method], 
                                imb_method)()
            runner = Runner(ordinal_label=label, 
                            epoch=epoch, data_method=norm, 
                            imb_class=imb_class, 
                            imb_text=imb_method)
            
            dict_metrics = runner.run()
            class_metrics[imb_method] = dict_metrics
            hyper_text = "classifier_default_methods_" + "_".join(class_undersampling_methods)
            folder_name = "MLP" + "_U2R_" + hyper_text + "_" + norm
            plot_2_pc_results(dataset_x=runner.X_dict, dataset_y=runner.y_dict,
                        res_dir=f"./res/address_imbalanced_res/{folder_name}")

        perf_text = f"original train distribution: {runner.orig_dist}\n\n"
        perf_text += generate_class_performance_text(class_metrics, imbalanced_problem=True)
        hyper_text = "classifier_default_methods_" + "_".join(class_undersampling_methods)

        save_result_text(classifier="MLP" + "_U2R_", 
                        hyper=hyper_text,
                        data_method=norm,
                        class_performance_text=perf_text,
                        )

        class_metrics = {}
        print("Oversampling methods\n----------------\n")
        for sample in sampling_strategies:
            print(f"---Testing with sampling strategy \"{sample}\"")
            for imb_method in class_oversampling_methods:
                print(f"---Testing with \"{imb_method}\" imbalance method---")
                imb_class = None
                if imb_method[0] == "S":
                    # oversampling
                    imb_class= getattr(class_imbalanced_methods_mapping[imb_method], 
                                        imb_method) \
                                        (sampling_strategy=sample, 
                                        random_state=seed)       
                else:
                    # under-sampling
                    imb_class= getattr(class_imbalanced_methods_mapping[imb_method], 
                                        imb_method)()
                runner = Runner(ordinal_label=label, 
                                epoch=epoch, data_method=norm, 
                                imb_class=imb_class, 
                                imb_text=imb_method)
                
                dict_metrics = runner.run()
                class_metrics[imb_method] = dict_metrics
                hyper_text = "classifier_default_methods_" + "_".join(class_oversampling_methods)
                hyper_text += f"_sampling_strategy_{sample}"
                folder_name = "MLP" + "_U2R_" + hyper_text + "_" + norm
                plot_2_pc_results(dataset_x=runner.X_dict, dataset_y=runner.y_dict,
                            res_dir=f"./res/address_imbalanced_res/{folder_name}")

            perf_text = f"original train distribution: {runner.orig_dist}\n\n"
            perf_text += generate_class_performance_text(class_metrics, imbalanced_problem=True)
            hyper_text = "classifier_default_methods_" + "_".join(class_oversampling_methods)
            hyper_text += f"_sampling_strategy_{sample}"

            save_result_text(classifier="MLP" + "_U2R_", 
                            hyper=hyper_text,
                            data_method=norm,
                            class_performance_text=perf_text,
                            )

    

    end = time.time()
    print(f"\nFinished training in {(end - start):.0f}s")


if __name__ == '__main__':
    main()
    

