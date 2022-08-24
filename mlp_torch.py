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
    def __init__(self, model, optimizer, loss_fn, epoch=1, batch_size=128, ordinal_label=0):
        self.ordinal_label = ordinal_label
        self.ds = MyDataset(ordinal_label=0)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def run(self):
        folds = get_train_test_indices_for_all_folds(self.ds.get_df())
        start = time.time()
        for fold, (train_idx, test_idx) in enumerate(folds):
            print(f'---fold no----{fold+1}---')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

            train_loader = DataLoader(
                                self.ds, 
                                batch_size=self.batch_size,
                                sampler=train_subsampler)
            test_loader = DataLoader(
                                self.ds,
                                batch_size=len(test_idx), 
                                sampler=test_subsampler
                                )

            # model.apply(reset_weights)

            for epoch in range(self.epoch):
                print(f'---epoch no---{epoch+1}---')
                self.train(train_loader)
                self.test(epoch, test_loader, test_idx)

        end = time.time()
        
        target, pred = self.test(fold, test_loader, test_idx, val=False)
        if self.device.type == 'cuda':
            target, pred = target.cpu(), pred.cpu()
        class_metric = {}
        dict_metric = compute_metric_values(target.detach().numpy(), pred.detach().numpy())
        dict_metric["Training time"] = round(end - start, 2)
        class_name = self.ds.get_text_label()
        class_metric[class_name] = dict_metric
        perf_text = generate_class_performance_text(class_metric)
        save_result_text(classifier="MLP", 
                        hyper="hidden_layer_80_100", 
                        data_method="minmax",
                        class_performance_text=perf_text)


    def train(self, train_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

    def test(self, epoch, test_loader, test_idx, val=True):
        self.model.eval()
        test_loss = 0
        correct = 0
        output = None
        target = None
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_fn(output, target, reduction='sum').item()  
                pred = torch.where(output >= 0.5, 1, 0) 
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_idx)
        text = ""
        text += f'\nTest set for epoch {epoch+1}: Average loss: {test_loss:.4f}, '
        text += f'Accuracy: {correct}/{len(test_idx)} ({ 100. * correct / len(test_idx):.0f}%)\n'
        print(text)

        if not val:
            return torch.flatten(target), torch.flatten(pred)

        
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
    model = MLP()
    runner = Runner(model=model, 
                    optimizer=torch.optim.Adam(params=model.parameters()),
                    loss_fn=F.binary_cross_entropy,
                    ordinal_label=1)
    runner.run()

if __name__ == '__main__':
    main()
    

