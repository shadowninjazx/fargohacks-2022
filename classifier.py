import pandas
import torch
import torch.nn as nn
import re
from tqdm import tqdm
import os
import time
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransactionDataset(torch.utils.data.Dataset):
    '''
    TransactionDataset
    A custom Pytorch dataset for handling of feature-label pairs. 

    Args:
        feature_list - a list of feature vectors
        label_list - a list of one-hot-encoded labels
    '''
    def __init__(self, feature_list, label_list=None):
        super().__init__()
        self.data = feature_list
        self.label = label_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]


class NNClassifier(nn.Module):
    '''
    NNClassifier
    A Pytorch Module fully connected classifier

    Args:
        dim_features - number of input features
        num_classes - number of output classes
        dim_hidden - number of hidden dimensions
        num_layers - number of hidden layers
    '''
    def __init__(self, dim_features, num_classes=10, dim_hidden=256, num_layers=1):
        super().__init__()

        net_list = [nn.Linear(dim_features, dim_hidden), nn.ReLU()]
        for i in range(num_layers):
            net_list.append(nn.Linear(dim_hidden, dim_hidden))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(dim_hidden, num_classes))

        self.net = nn.Sequential(*net_list)

    def forward(self, x):
        return self.net(x)


class TransactionClassifier:
    '''
    TransactionClassifier
    A custom class for the Campus Analytics Challenge 2022

    Args:
        n/a
    '''
    def __init__(self):

        # PREDEFINED LIST OF CATEGORIES
        self.category_list = ['Finance', 'Services to Transport', 'Communication Services',
                            'Property and Business Services', 'Travel', 'Entertainment', 'Education',
                            'Health and Community Services',
                            'Trade, Professional and Personal Services', 'Retail Trade']

        # PREDEFINED CATEGORY LEVEL DATASET REBALANCING 
        self.class_mult = {'Finance':4, 'Services to Transport':1, 'Communication Services':3,
                            'Property and Business Services':1, 'Travel':1, 'Entertainment':1, 'Education':2,
                            'Health and Community Services':1,
                            'Trade, Professional and Personal Services':1, 'Retail Trade':1}

        # INITIALIZING HELPER DICTIONARIES
        self.cat_to_idx = {}
        self.idx_to_cat = {}
        for i in range(len(self.category_list)):
            self.cat_to_idx[self.category_list[i]] = i
            self.idx_to_cat[i] = self.category_list[i]

    def load_features(self, train_features_ckpt, layer, use_one_hot, train=True):
        '''
        load_features
        loads features from a Pytorch checkpoint

        Args:
            train_features_ckpt - the address of feature checkpoint
            layer - the intermediate layer to extract the features from (used for experiments) (Default:None)
            use_one_hot - whether or not to include one_hot features 
            train - whether or not to return labels
        Returns:
            X - (N, F) a tensor of training features
            y - (if train=True) a (N, 10) tensor of training labels
        '''

        # LOAD PYTORCH CHECKPOINT
        start = time.time()
        train_features = torch.load(train_features_ckpt)
        print('loaded', len(train_features), 'training features from', train_features_ckpt, 'in', time.time()-start, 'seconds')
        
        # ITERATE THROUGH ALL ENTRIES
        x_list = []
        x_one_hot_list = []
        y_list = []
        for i, item in enumerate(train_features):
            # APPEND FEATURES TO LISTS
            if train:
                for mult in range(self.class_mult[item['category']]):
                    if layer is None:
                        x_list.append(item['features'][0])
                    else:
                        x_list.append(item['features'][layer][0])
                    if use_one_hot:
                        x_one_hot_list.append(item['one_hot'][0])
                    y_list.append(torch.tensor(self.cat_to_idx[item['category']]))
            else:
                if layer is None:
                    x_list.append(item['features'][0])
                else:
                    x_list.append(item['features'][layer][0])
                if use_one_hot:
                        x_one_hot_list.append(item['one_hot'][0])
       
        # CONVER LISTS TO TENSORS
        x_one_hot = torch.stack(x_one_hot_list)
        X = torch.stack(x_list)
        X = torch.cat((X, x_one_hot), dim=-1)
        if not train:
            print(X.shape)
            return X
        
        y = torch.stack(y_list).long()
        print(X.shape, y.shape)
        return X, y


    def train(self, train_features_ckpt, save_addr='ckpt/model.pt', max_epoch=50, split_ratio=0.6, seed=0, layer=None, use_one_hot=True, dim_hidden=1024):
        '''
        train
        training loop

        Args:
            train_feature_ckpt - the addres of feature checkpoint 
            save_addr - address to save the model weights
            max_epochs - number of epochs to train
            split_ratio - ratio of the training set
            seed - random seed
            layer - layer of feature to use (used for experiments) (default: None)
            use_one_hot - whether or not to use one-hot features (default: True)
            dim_hidden - number of hidden dims in classifier
        '''
        # LOAD TRAIN FEATURES
        X, y = self.load_features(train_features_ckpt, layer, use_one_hot)
        
        # INIT MODEL
        dim_features = X.shape[-1]
        net = NNClassifier(dim_features=dim_features, dim_hidden=dim_hidden).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        # SPLIT TRAIN AND VAL SETS
        torch.manual_seed(seed)
        rand = torch.randperm(len(X))
        train_ratio = split_ratio
        
        print(int(train_ratio*len(rand)))
        train_set = TransactionDataset(X[rand[:int(train_ratio*len(rand))]], y[rand[:int(train_ratio*len(rand))]])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=4)
        test_set = TransactionDataset(X[rand[int(train_ratio*len(rand)):]], y[rand[int(train_ratio*len(rand)):]])
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=True, num_workers=4)

        # TRAINING LOOP
        for e in tqdm(range(max_epoch)):
            running_correct, running_count = 0, 0
            confusion_matrix = np.zeros((len(self.category_list), len(self.category_list)))

            net = net.train()

            #! TRAIN
            for x, y in (train_loader):
                x = x.to(device)
                y = y.to(device)
                logits = net(x)
                loss = criterion(logits, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                y_hat = torch.argmax(logits, dim=1)
                running_correct += (y_hat == y).sum().item()
                running_count += len(y)

            #! VAL
            val_correct, val_count = 0, 0 
            with torch.no_grad():
                net = net.eval()
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = net(x)
                    y_hat = torch.argmax(logits, dim=1)
                    val_correct += (y_hat == y).sum().item()
                    val_count += len(y)

                    for pred_, true_ in zip(y_hat.cpu().numpy().flatten(), y.cpu().numpy().flatten()):
                        confusion_matrix[int(true_), int(pred_)] += 1


            train_acc = running_correct / running_count
            val_acc = val_correct / val_count

        # PLOT CONFUSION MATRIX
        confusion_matrix=100*confusion_matrix/confusion_matrix.sum(1, keepdims=True)
        df_cm = pd.DataFrame(confusion_matrix, index = [i[:15] for i in self.category_list],
                            columns = [i[:15] for i in self.category_list]).astype(int)
        plt.figure(figsize = (12,9))
        sns.heatmap(df_cm, annot=True, fmt='d')
        plt.savefig('output.png', bbox_inches='tight')
        print(f'train:', train_acc, 'val:', val_acc)

        # SAVE MOEL
        torch.save({'model_state_dict':net.state_dict()}, save_addr)
        print('saved model to', save_addr)


    def save_test_predictions(self, model_ckpt, test_features_ckpt, test_csv='data/test.xlsx', out_csv='output/test_predictions.xlsx', layer=None, use_one_hot=True):
        '''
        save_test_predictions
        load test features and mode checkpoint and test csv and fill out the predictions

        Args:
            model_ckpt: checkpoint of Pytorch model
            test_features_ckpt: checkpoint of test features
            test_csv: address of test XLSX
            out_csv: address for the output xlsx files
            layer - layer of feature to use (used for experiments) (default: None)
            use_one_hot - whether or not to use one-hot features (default: True)
        '''
        
        # LOAD FEATURES
        X = self.load_features(test_features_ckpt, layer, use_one_hot, train=False)
        dim_features = X.shape[-1]

        # LOAD MODELS
        net = NNClassifier(dim_features=dim_features, dim_hidden=1024).to(device)
        ckpt = torch.load(model_ckpt)
        net.load_state_dict(ckpt['model_state_dict'])
        print('loaded model from', model_ckpt)

        # INIT DATASET
        final_test_set = TransactionDataset(X)

        # RUN PREDICTIONS
        prediction_dict = {}
        with torch.no_grad():
            for idx in tqdm(range(len(final_test_set))):
                x = final_test_set.__getitem__(idx)
                x = x.to(device)
                logits = net(x)
                y_hat = torch.argmax(logits, dim=0).cpu()
                y_hat_cat = self.idx_to_cat[int(y_hat)]
                prediction_dict[idx] = y_hat_cat

        # WRITE TO XLSX AND CSV
        df = pandas.read_excel(test_csv, engine='openpyxl')

        for item in prediction_dict:
            df.at[item,'Category']=prediction_dict[item]

        df.to_csv(out_csv.replace('.xlsx', '.csv'), index=False)
        df.to_excel(out_csv, index=False)  


if __name__ == '__main__':

    os.makedirs('ckpt', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    parser = argparse.ArgumentParser(description='Transaction Classifier')
    parser.add_argument('-f', '--train_features', type=str, default='data/train_features.pt', metavar='pt', required=True,
                        help='location of training features')
    parser.add_argument('-t', '--test_features', type=str, default=None, metavar='pt',
                        help='location of test features')
    parser.add_argument('-c', '--test_xlsx', type=str, default='data/test.xlsx', metavar='xlsx',
                        help='location of test xlsx')
    parser.add_argument('-o', '--output', type=str, default='output/test_predictions.xlsx', metavar='xlsx',
                        help='output location of predictions')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    model = TransactionClassifier()

    print('training...')
    model.train(args.train_features, max_epoch=50, split_ratio=0.8, seed=0, layer=None, use_one_hot=True, dim_hidden=1024)

    if args.test_features is not None:
        print('testing...')
        model.save_test_predictions('ckpt/model.pt', test_features_ckpt=args.test_features, test_csv='data/test.xlsx', layer=None, use_one_hot=True)
