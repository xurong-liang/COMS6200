"""
Implement the Multilayer Perception (MLP) Classifier using Pytorch
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
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

class MLP(nn.Module):
    def __init__(self, input_dim=58, output_dim=1):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 80)
        self.hidden_fc = nn.Linear(80, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = torch.sigmoid(self.output_fc(h_2))
        return torch.flatten(y_pred)

class MyDataset(Dataset):
    def __init__(self, ordinal_label):
        self.df, self.onehot, self.label2, self.label3, self.label4 = get_data_frame()
        self.df['ordinal_label'] = self.df['ordinal_label'].apply(lambda x: 1 if x == ordinal_label else 0)
        X = self.df.drop(columns = ['text_label', 'ordinal_label']).values
        y = self.onehot[:, ordinal_label]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ordinal_label = ordinal_label
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_df(self):
        return self.df

    def get_text_label(self):
        return self.label4[self.ordinal_label][0]

    def get_onehot_label(self):
        return self.label4[self.ordinal_label][1]


class Runner():
    def __init__(self, epoch=2, batch_size=128, ordinal_label=0):
        self.ordinal_label = ordinal_label
        self.ds = MyDataset(ordinal_label=0)
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run(self):
        folds = get_train_test_indices_for_all_folds(self.ds.get_df(), k=2)
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
                metrics_dict = self.test(epoch, test_loader, model, loss_fn, test_idx)
                result_folds.append(metrics_dict)

        end = time.time()
            #     {
            #     "accuracy": acc,
            #     "precision": precision,
            #     'recall': recall,
            #     "f1": f1
            #      }

            # {
            #     "normal": {
            #                 "accuracy": acc,
            #                 "precision": precision,
            #                 'recall': recall,
            #                 "f1": f1,
            #                 "training time": training time
            #             }
            # }

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

        class_name = self.ds.get_text_label()
        class_metrics = {}
        class_metrics[class_name] = dict_metrics
        perf_text = generate_class_performance_text(class_metrics)
        save_result_text(classifier="MLP", 
                        hyper="hidden_layer_80_100", 
                        data_method="minmax",
                        class_performance_text=perf_text)


    def train(self, train_loader, model, optimizer, loss_fn):
        model.train()
        for (data, target) in tqdm(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    def test(self, epoch, test_loader, model, loss_fn, test_idx):
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
        text += f'\nTest set for epoch {epoch+1}: Average loss: {loss:.4f}, '
        text += f'Accuracy: {correct}/{len(test_idx)} ({ correct / len(test_idx):.8f})\n'
        print(text)

        if self.device.type == 'cuda':
            target, pred = target.cpu(), pred.cpu()
        metrics_dict = compute_metric_values(target.detach().numpy(), 
                                             pred.detach().numpy())
        return metrics_dict

        
def main():
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
    runner = Runner(ordinal_label=1)
    runner.run()

if __name__ == '__main__':
    main()
    

