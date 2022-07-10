from numpy import require
import pandas
import torch
import re
from tqdm import tqdm
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CategoricalEncoding:
    '''
    CategoricalEncoding
    A class for categorical one-hot-encoding

    Args:
        initialization_list: a list of words to process into the categorical dictionary
    '''
    def __init__(self, iniitalization_list):
        self.dict = {}
        for item in iniitalization_list:
            item = str(item)
            if item is not None and item != '' and item != 'nan':
                if item in self.dict:
                    continue
                else:
                    self.dict[item] = len(self.dict)
        print('one hot encoding with', len(self.dict), 'dictionary entries')
    
    def encode(self, item):
        feature = torch.zeros((len(self.dict)))
        if item is not None and item != '' and item != 'nan':
            if item in self.dict:
                feature[self.dict[item]] = 1
                return feature
            else:
                return feature
        else:
            return feature


def get_encoder(model_size='large'):
    '''
    A function to get get ROBERTa text encoder from Huggingface

    Args:
        model_size: either base or large
    '''
    assert model_size == 'base' or model_size == 'large'
    from transformers import RobertaTokenizer, RobertaModel
    tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')
    model = RobertaModel.from_pretrained(f'roberta-{model_size}').to(device)
    return tokenizer, model


def get_categorical_encoder(csv_file_name, categorical_variable):
    '''
    A function to get categorical encoder

    Args:
        csv_file_name: training xlsx filre
        categorical_variable: name of categorical variable to encode
    '''
    df = pandas.read_excel(csv_file_name)
    categorical_encoder = CategoricalEncoding(df[categorical_variable].tolist())
    return categorical_encoder


def load_dataset(csv_file_name, text_variables, categorical_variables=None, categorical_encoders=None):
    '''
    Reads xlsx and outputs a list of context, category pairs
    Args:
        csv_file_name: name of xlsx
    Returns:
        text_list: a list of text
        categorical_features: a list of one-hot categorical features
    '''

    # READ XLSX
    df = pandas.read_excel(csv_file_name)
    print('read', csv_file_name, 'and found', len(df), 'rows')
    print('unique categories', df["Category"].unique())

    # LOOP THROUGH ROWS
    text_list = []
    one_hot_list = []
    for index, row in df.iterrows():
        category = row["Category"]

        #! CONCATENATE TEXT VARIABLES
        tmp_text = []
        for tv in text_variables:
            tmp_text.append(str(row[tv]))
        context = ' '.join(tmp_text)
        text_list.append({'index':index, 'context':context, 'category':category})

        #! CONCATENATE CATEGORICAL VARIABLES
        if categorical_variables is not None:
            tmp_one_hot = []
            for cv, ce in zip(categorical_variables, categorical_encoders):
                tmp_one_hot.append(ce.encode(str(row[cv])))
            one_hot = torch.stack(tmp_one_hot)
            one_hot_list.append(one_hot)
    
    return text_list, one_hot_list


def plot_histogram(csv_file_name, output='count.png'):
    '''
    A function to plot category histogram

    Args:
        csv_file_name: address of xlsx
        output: output image name
    '''
    df = pandas.read_excel(csv_file_name)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize = (12,7))
    countplot = sns.countplot(x='Category', data=df)
    countplot.xaxis.set_ticklabels(countplot.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=10)
    countplot.set_xlabel("Categories",fontsize=15)
    countplot.set_ylabel("Count",fontsize=15)
    countplot.bar_label(countplot.containers[0])
    fig = countplot.get_figure()
    plt.savefig(output, bbox_inches='tight')


def extract_features(text_list, one_hot_list, tokenizer, model, output_hidden_states=True):
    '''
    A function that extracts features from list of text and one_hot variables

    Args:
        text_list: a list of text
        one_hot_list: a list of one hot vectors
        tokenizer: ROBERTa tokenizer
        model: ROBERTa model
        output_hidden_states: whether to extract intermediate features

    Returns:
        feature_list: a list of feature lable pairs
    '''
    feature_list = []
    i_ = 0
    for item in tqdm(text_list):
        index = item['index']
        context = item['context']
        category = item['category']

        try:
            encoded_input = tokenizer(context, return_tensors='pt').to(device)
        except:
            encoded_input = tokenizer('null', return_tensors='pt').to(device)

        output = model(**encoded_input, output_hidden_states=output_hidden_states)

        if output_hidden_states:
            hidden_state = output.hidden_states
            hidden = []
            for i, s in enumerate(hidden_state):
                hidden.append(s.detach().cpu().mean(dim=1))
            features = output.last_hidden_state.detach().cpu()
            avg_features = features.mean(dim=1)
            feature_list.append({'index':index, 'context':context, 'category':category, 'features':avg_features, 'hidden':hidden, 'one_hot':one_hot_list[i_]})
        else:
            features = output.last_hidden_state.detach().cpu()
            avg_features = features.mean(dim=1)
            feature_list.append({'index':index, 'context':context, 'category':category, 'features':avg_features, 'one_hot':one_hot_list[i_]})
        i_ += 1
    return feature_list


def save_features_pt(features, save_addr, hidden_layer=13):
    '''
    A function that takes feature_list and saves it as feature checkpoint
    
    Args:
        features: the feature list from extract_features
        save_addr: the place to save the feature checkpoint
        hidden_layer: the intermediate layer to use as text features
    '''
    feature_list = []
    print('saving', len(features), 'features')
    for item in features:
        if hidden_layer == 'all':
            feature_list.append({'index':item['index'], 'context':item['context'], 'category':item['category'], 'features':item['hidden'], 'one_hot':item['one_hot']})
        elif hidden_layer is not None:
            feature_list.append({'index':item['index'], 'context':item['context'], 'category':item['category'], 'features':item['hidden'][hidden_layer], 'one_hot':item['one_hot']})
    torch.save(feature_list, f'{save_addr}')
    print('saved to', save_addr)


def save_train_and_test_features(train_file, test_file, text_variables, categorical_variables, model_size, hidden_layer, mod=''):
    '''
    A helper function to save train and test features checkpoints

    Args:
        train_file: training xlsx
        test_file: test xlsx
        text_variables: a list of variables to encode with text encoder
        categorical_variables: a list of variables to encode with categorical encoder
        model_size: size of ROBERTa model
        hidden_layer: the intermediate layer of ROBERTa to use as text features
        mod: suffix for saved checkpoints
    '''

    #! GET ENCODERS
    tokenizer, roberta = get_encoder(model_size=model_size)
    categorical_encoders = []
    for cv in categorical_variables:
        categorical_encoders.append(get_categorical_encoder(train_file, cv))

    #! READ AND SAVE TRAINING DATA
    train_text_list, train_one_hot_list = load_dataset(train_file, text_variables=text_variables, categorical_variables=categorical_variables, categorical_encoders=categorical_encoders)
    train_features = extract_features(train_text_list, train_one_hot_list, tokenizer, roberta, output_hidden_states=True)
    save_features_pt(train_features, f'data/train_features_{mod}_{model_size}_l{hidden_layer}.pt', hidden_layer=hidden_layer)


    #! READ AND SAVE TRAINING DATA
    test_text_list, test_one_hot_list = load_dataset(test_file, text_variables=text_variables, categorical_variables=categorical_variables, categorical_encoders=categorical_encoders)
    test_features = extract_features(test_text_list, test_one_hot_list, tokenizer, roberta, output_hidden_states=True)
    save_features_pt(test_features, f'data/test_features_{mod}_{model_size}_l{hidden_layer}.pt', hidden_layer=hidden_layer)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Feature Extractor')
    parser.add_argument('-f', '--train_xlsx', type=str, default='data/train.xlsx', metavar='xlsx', required=True,
                        help='location of train xlsx')
    parser.add_argument('-t', '--test_xlsx', type=str, default='data/test.xlsx', metavar='xlsx', required=True,
                        help='location of test xlsx')
    args = parser.parse_args()

    #! LOAD ROBERTa Model
    mod = 'bm'                                       # save suffix
    text_variables = ['coalesced_brand']             # text variables
    categorical_variables = ['merchant_cat_code']    # categorical variables
    model_size = 'large'                             # large outperforms base by 2 percentage points
    hidden_layer = 13                                # 13 for large model; 7 for base model
    save_train_and_test_features(args.train_xlsx, args.test_xlsx, text_variables, categorical_variables, model_size, hidden_layer, mod=mod)
