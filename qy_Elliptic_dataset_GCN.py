#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/JungWoo-Chae/GCN_Elliptic_dataset/blob/main/Elliptic_dataset_GCN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Bitcoin Fraud Detection System with GCN

# ## Pytorch Geometric Environment Setting

# In[ ]:


# Install required packages.
# get_ipython().system('pip install -q torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html')
# get_ipython().system('pip install -q torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html')
# get_ipython().system('pip install -q git+https://github.com/rusty1s/pytorch_geometric.git')


# ## Library Import

# In[9]:


import numpy as np
import networkx as nx
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import Parameter
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric_temporal.nn.recurrent import EvolveGCNO

from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_undirected

from modle_GCN2layer import GCN as GCN2layer
from try_EGCNO import RecurrentGCN as EGCNO

CHOOSE_MODE = {
    "GCN": GCN2layer,
    "EGCN_H": EGCNO
}
# In[10]:
use_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(use_dev)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # **Please insert Kaggle username and kaggle key**

# In[ ]:


# os.environ['KAGGLE_USERNAME'] = "@@@@@@@@@" # username from the json file
# os.environ['KAGGLE_KEY'] = "####################" # key from the json file
# get_ipython().system('kaggle datasets download -d ellipticco/elliptic-data-set')
# get_ipython().system('unzip elliptic-data-set.zip')
# get_ipython().system('mkdir elliptic_bitcoin_dataset_cont')


# ## Data Preparation

# In[ ]:


def make_data(tag="mkdat"):
    if os.path.exists('dat_' + f'{tag}'):
        with open('dat_' + f'{tag}', "rb") as f:
            ALL_DATA = pickle.load(f)
        return ALL_DATA

    # Load Dataframe
    df_edge = pd.read_csv('elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    df_class = pd.read_csv('elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    df_features = pd.read_csv('elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)

    # Setting Column name
    df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in
                                                                                          range(72)]

    print('Number of edges: {}'.format(len(df_edge)))

    # ## Get Node Index

    # In[ ]:

    all_nodes = list(
        set(df_edge['txId1']).union(set(df_edge['txId2'])).union(set(df_class['txId'])).union(set(df_features['id'])))
    nodes_df = pd.DataFrame(all_nodes, columns=['id']).reset_index()
    NUM_NODES = len(nodes_df)
    print('Number of nodes: {}'.format(NUM_NODES))
    NUM_FEAT = 166
    # ## Fix id index

    # In[ ]:

    df_edge = df_edge.join(nodes_df.rename(columns={'id': 'txId1'}).set_index('txId1'), on='txId1', how='inner').join(
        nodes_df.rename(columns={'id': 'txId2'}).set_index('txId2'), on='txId2', how='inner', rsuffix='2').drop(
        columns=['txId1', 'txId2']).rename(columns={'index': 'txId1', 'index2': 'txId2'})
    df_edge.head()

    # In[ ]:

    df_class = df_class.join(nodes_df.rename(columns={'id': 'txId'}).set_index('txId'), on='txId', how='inner').drop(
        columns=['txId']).rename(columns={'index': 'txId'})[['txId', 'class']]
    df_class.head()

    # In[ ]:

    df_features = df_features.join(nodes_df.set_index('id'), on='id', how='inner').drop(columns=['id']).rename(
        columns={'index': 'id'})
    df_features = df_features[['id'] + list(df_features.drop(columns=['id']).columns)]
    df_features.head()

    # In[ ]:

    df_edge_time = df_edge.join(df_features[['id', 'time step']].rename(columns={'id': 'txId1'}).set_index('txId1'),
                                on='txId1', how='left', rsuffix='1').join(
        df_features[['id', 'time step']].rename(columns={'id': 'txId2'}).set_index('txId2'), on='txId2', how='left',
        rsuffix='2')
    df_edge_time['is_time_same'] = df_edge_time['time step'] == df_edge_time['time step2']
    df_edge_time_fin = df_edge_time[['txId1', 'txId2', 'time step']].rename(
        columns={'txId1': 'source', 'txId2': 'target', 'time step': 'time'})

    # ## Create csv from Dataframe

    # In[ ]:

    df_features.drop(columns=['time step']).to_csv('elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv',
                                                   index=False,
                                                   header=None)
    df_class.rename(columns={'txId': 'nid', 'class': 'label'})[['nid', 'label']].sort_values(by='nid').to_csv(
        'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv', index=False, header=None)
    df_features[['id', 'time step']].rename(columns={'id': 'nid', 'time step': 'time'})[['nid', 'time']].sort_values(
        by='nid').to_csv('elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv', index=False, header=None)
    df_edge_time_fin[['source', 'target', 'time']].to_csv(
        'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv',
        index=False, header=None)

    # ## Graph Preprocessing

    # In[ ]:

    node_label = df_class.rename(columns={'txId': 'nid', 'class': 'label'})[['nid', 'label']].sort_values(
        by='nid').merge(
        df_features[['id', 'time step']].rename(columns={'id': 'nid', 'time step': 'time'}), on='nid', how='left')
    node_label['label'] = node_label['label'].apply(lambda x: '3' if x == 'unknown' else x).astype(int) - 1
    node_label.head()

    # In[ ]:

    merged_nodes_df = node_label.merge(
        df_features.rename(columns={'id': 'nid', 'time step': 'time'}).drop(columns=['time']), on='nid', how='left')
    merged_nodes_df.head()

    # In[ ]:

    train_dataset = []
    test_dataset = []
    test_dataset2 = []
    for i in range(49):
        nodes_df_tmp = merged_nodes_df[merged_nodes_df['time'] == i + 1].reset_index()
        nodes_df_tmp['index'] = nodes_df_tmp.index
        df_edge_tmp = df_edge_time_fin.join(
            nodes_df_tmp.rename(columns={'nid': 'source'})[['source', 'index']].set_index('source'), on='source',
            how='inner').join(nodes_df_tmp.rename(columns={'nid': 'target'})[['target', 'index']].set_index('target'),
                              on='target', how='inner', rsuffix='2').drop(columns=['source', 'target']).rename(
            columns={'index': 'source', 'index2': 'target'})
        x = torch.tensor(np.array(nodes_df_tmp.sort_values(by='index').drop(columns=['index', 'nid', 'label'])),
                         dtype=torch.float)
        edge_index = torch.tensor(np.array(df_edge_tmp[['source', 'target']]).T, dtype=torch.long)
        edge_index = to_undirected(edge_index)
        mask = nodes_df_tmp['label'] != 2
        y = torch.tensor(np.array(nodes_df_tmp['label']))
        # 划分测试集
        if i + 1 < 35:
            data = Data(x=x, edge_index=edge_index, train_mask=mask, y=y)
            train_dataset.append(data)
        else:
            data = Data(x=x, edge_index=edge_index, test_mask=mask, y=y)
            test_dataset.append(data)
        # elif i+1 < 35+5:
        #     data = Data(x=x, edge_index=edge_index, test_mask=mask, y=y)
        #     test_dataset.append(data)
        # else:
        #     data = Data(x=x, edge_index=edge_index, test_mask=mask, y=y)
        #     test_dataset2.append(data)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)
    ALL_DATA = [NUM_NODES, NUM_FEAT, train_loader, test_loader]
    pickle.dump(ALL_DATA, open('dat_' + f'{tag}', 'wb'))
    with open('dat_' + f'{tag}', "rb") as f:
        ALL_DATA = pickle.load(f)
    return ALL_DATA


# ## Model

# In[ ]:

# In[ ]:


# ## Train

# #### Hyperparameter

# In[ ]:

def tain_model(train_loader, test_loader, model, use_criterion, epoches=1000, tag=""):
    patience = 50
    lr = 0.001
    epoches = epoches

    # In[ ]:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = use_criterion
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]).to(device))
    # criterion = F.nll_loss
    train_losses = []
    val_losses = []
    accuracies = []
    if1 = []
    precisions = []
    recalls = []
    iterations = []
    logf = f"log_{tag}"
    all_logstr=""
    for epoch in range(epoches):

        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask].long())
            _, pred = out[data.train_mask].max(dim=1)
            loss.backward()
            train_loss += loss.item() * data.num_graphs
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        # 评估验证集
        if (epoch + 1) % 50 == 0:
            model.eval()
            ys, preds = [], []
            val_loss = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out[data.test_mask], data.y[data.test_mask].long())
                val_loss += loss.item() * data.num_graphs
                _, pred = out[data.test_mask].max(dim=1)
                ys.append(data.y[data.test_mask].cpu())
                preds.append(pred.cpu())

            y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
            val_loss /= len(test_loader.dataset)
            f1 = f1_score(y, pred, average=None)
            mf1 = f1_score(y, pred, average='micro')
            precision = precision_score(y, pred, average=None)
            recall = recall_score(y, pred, average=None)

            iterations.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if1.append(f1[0])
            accuracies.append(mf1)
            precisions.append(precision[0])
            recalls.append(recall[0])
            log_str = 'Epoch: {:02d}, Train_Loss: {:.4f}, Val_Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Illicit f1: {:.4f}, mF1: {:.4f}'.format(
                    epoch + 1, train_loss, val_loss, precision[0], recall[0], f1[0], mf1) + "\n" +\
                f'T class precision: {precision[1]}, recall: {recall[1]}, f1: {f1[1]}\n'
            print(log_str
            )
            all_logstr+=log_str

    # In[ ]:
    with open(logf, "w+") as f:
        f.write(all_logstr)

    a, b, c, d = train_losses, val_losses, if1, accuracies

    import pickle

    g = [a, b, c, d]
    pickle.dump(g, open('res_' + f'{tag}', 'wb'))
    with open('res_' + f'{tag}', "rb") as f:
        g = pickle.load(f)
    a, b, c, d = g

    ep = [i for i in range(patience, epoches + 1, patience)]
    plt.figure()
    plt.plot(np.array(ep), np.array(a), 'r', label='Train loss')
    plt.plot(np.array(ep), np.array(b), 'g', label='Valid loss')
    plt.plot(np.array(ep), np.array(c), 'black', label='Illicit F1')
    plt.plot(np.array(ep), np.array(d), 'orange', label='mF1')
    plt.legend(['Train loss', 'Valid loss', 'Illicit F1', 'mF1'])
    plt.ylim([0, 1.0])
    plt.xlim([patience, epoches])
    plt.savefig(f"pic_{tag}.png")


if __name__ == "__main__":
    # In[ ]:
    # GCNN = EGCNO
    NUM_NODES, NUM_FEAT, train_loader, test_loader = make_data()
    epoches = 1000
    print(f"make_data ok!!!, epoches = {epoches}")
    for i in [("GCN_GCN", GCNConv, GCNConv, True), ("GAT_GAT", GATConv, GATConv, True),
              ("GCN_GAT", GCNConv, GATConv, True), ("GAT_GCN", GATConv, GCNConv, True)]:

        print(i[0])
        tag, conv1, conv2, useskip = i
        tag = tag + "_skip" + str(useskip)
        model = GCN2layer(NUM_FEAT, [100], conv1, conv2, use_skip=useskip)
        model.to(device)
        lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.9, 0.1]).to(device))
        tain_model(train_loader, test_loader, model, lossf, epoches=epoches, tag=tag)
        torch.save(model.state_dict(), f'model_{tag}.pkl')
        # break
        # 加载
        # model = torch.load(f'\model_{tag}.pkl')
        # model.load_state_dict(torch.load('\parameter.pkl'))

