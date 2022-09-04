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

    :param ordinal_label: Numerical label (between 0 and 7 in InSDN dataset)
    :param data_method: the dataset to be imported, options: [minmax, unnormalized, zscore] 
    """
    def __init__(self, ordinal_label, data_method="minmax"):
        df, onehot, self.label2, self.label3, self.label4 = get_data_frame(data_method=data_method)
        self.folds = get_train_test_indices_for_all_folds(df, k=3)
        # label with the correct attack = 1, otherwise 0 
        # ordinal label correspond to the column in onehot vector
        y = onehot[:, ordinal_label].astype(int)

        if ordinal_label == 0:
            y ^= 1 # flipping 0 and 1 since onehot normal = 1, need to convert to 0
            
        X = df.drop(columns = ['text_label', 'ordinal_label']).values
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ordinal_label = ordinal_label
        

    def __len__(self):
        return len(self.y)
    

    def get_folds(self):
        """
        Get the train and test fold indices
        """
        return self.folds


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


    def get_text_label(self):
        """
        Get the text label, derived from ordinal label
        """
        return self.label4[self.ordinal_label][0]


    def get_onehot_label(self):
        """
        Get the one-hot label, derived from ordinal label
        """
        return self.label4[self.ordinal_label][1]


class Runner():
    """
    Runner class, handle training and testing and saving results

    :param epoch: Number of epoch in training
    :param batch_size: Number of datapoints in a batch
    :param ordinal_label: Numerical label (between 0 and 7 in InSDN dataset)
    :param data_method: the dataset to be imported, options: [minmax, unnormalized, zscore] 
    """
    def __init__(self, epoch=10, batch_size=128, ordinal_label=0, data_method="minmax"):
        self.ds = MyDataset(ordinal_label, data_method)
        self.text_label = self.ds.get_text_label()
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

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
        folds = self.ds.get_folds()
        start = time.time()
        result_folds = []
        for fold, (train_idx, test_idx) in enumerate(folds):
            print(f'---fold no----{fold+1}---')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

            train_loader = DataLoader(
                                self.ds, 
                                batch_size=self.batch_size,
                                sampler=train_subsampler)

            # faster to test on a single batch of every test data points
            test_loader = DataLoader(
                                self.ds,
                                batch_size=len(test_idx), 
                                sampler=test_subsampler
                                )
            model = MLP()
            model.to(self.device)
            model.apply(reset_weights)
            optimizer = torch.optim.Adam(params=model.parameters())
            loss_fn=F.binary_cross_entropy

            for epoch in range(self.epoch):
                print(f'---epoch no---{epoch+1}---')
                self.train(train_loader, model, optimizer, loss_fn)

            metrics_dict = self.test(fold, test_loader, model, loss_fn, test_idx)
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

        dict_metrics["Training time"] = round(end - start, 2)
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

    def test(self, fold, test_loader, model, loss_fn, test_idx):
        """
        Model testing function

        :param fold: number of fold
        :param test_loader: pytorch test data loader
        :param model: pytorch ML model
        :param optimizer: pytorch optimizer function
        :param loss_fn: pytorch loss function
        :param test_idx: test indices list (for getting length of test data)
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

        loss /= len(test_idx)
        text = ""
        text += f'\nTest set for fold {fold+1}: Average loss: {loss:.4f}, '
        text += f'Accuracy: {correct}/{len(test_idx)} ({ correct / len(test_idx):.8f})\n'
        print(text)

        if self.device.type == 'cuda':
            target, pred = target.cpu(), pred.cpu()
        metrics_dict = compute_metric_values(target.detach().numpy(), 
                                             pred.detach().numpy())
        return metrics_dict

        
def main():
    """
    Main function of MLP training model in InSDN project
    """

    """
    {'BOTNET': (7, array([0., 0., 0., 0., 0., 0., 0., 1.])),
    'Web-Attack': (6, array([0., 0., 0., 0., 0., 0., 1., 0.])),
    'Probe': (5, array([0., 0., 0., 0., 0., 1., 0., 0.])),
    'DoS': (4, array([0., 0., 0., 0., 1., 0., 0., 0.])),
    'DDoS': (3, array([0., 0., 0., 1., 0., 0., 0., 0.])),
    'BFA': (2, array([0., 0., 1., 0., 0., 0., 0., 0.])),
    'U2R': (1, array([0., 1., 0., 0., 0., 0., 0., 0.])),
    'Normal': (0, array([1., 0., 0., 0., 0., 0., 0., 0.]))}
    """
    start = time.time()
    for norm in ["minmax", "unnormalized", "zscore"]:
        print(f"---Testing with \"{norm}\" norm method---")
        class_metrics = {}
        for label in range(8):
            runner = Runner(ordinal_label=label, epoch=10, data_method=norm)
            text_label = runner.get_text_label()
            print(f"---Testing with label \"{text_label}\"")
            dict_metrics = runner.run()
            class_metrics[text_label] = dict_metrics

        perf_text = generate_class_performance_text(class_metrics)
        save_result_text(classifier="MLP", 
                        hyper="hidden_layer_80_100",
                        data_method=norm,
                        class_performance_text=perf_text)

    end = time.time()
    print(f"\nFinished training in {(end - start):.0f}s")


if __name__ == '__main__':
    main()
    

