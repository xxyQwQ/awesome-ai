from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def arxiv_dataset(transform=False):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data', transform=T.ToSparseTensor(layout=torch.sparse_coo) if transform else None)
    split_idx = dataset.get_idx_split()
    graph = dataset[0]
    return {
        'train_idx': split_idx['train'], # 90941
        'valid_idx': split_idx['valid'], # 29799
        'test_idx': split_idx['test'],   # 48603
        'graph': graph
        }

def load_titleabs(path='./data/titleabs.tsv'):
    data = pd.read_csv(path, sep='\t', header=None, names=['paper id', 'title', 'abs'])[1:-1] # Does't matter, the 0th row is not included in the graph
    nodeidx2paperid = pd.read_csv('./data/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
    data['paper id'] = data['paper id'].astype(int)
    return pd.merge(nodeidx2paperid, data, on='paper id')

def _concat_features(graph, emb_path='./data/embeddings_cls.pth', device='cuda:0'):
    '''
    concat bert embeddings, node year and node feat
    '''
    if emb_path is not None:
        bert_emb = torch.load(emb_path).to(device)
    else:
        bert_emb = torch.zeros((graph['num_nodes'], 768)).to(device)
    year = torch.tensor(graph['node_year']).to(device).type(torch.float32)
    original_features = torch.tensor(graph['node_feat']).to(device)
    return torch.cat([original_features, year, bert_emb], dim=1) # (169343, 897)

def get_neighbour_loader(batch_size=4096, num_workers=12, device='cuda:0'):
    '''
    return train_loader, test_loader using NeighborLoader. note that I add valid data into train data
    '''
    data = arxiv_dataset()
    train_loader = NeighborLoader(
            data['graph'],
            input_nodes=np.concatenate([data['train_idx'], data['valid_idx']]),
            num_neighbors=[10, 10],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    test_loader = NeighborLoader(
            data['graph'],
            input_nodes=data['test_idx'],
            num_neighbors=[10, 10],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    return train_loader, test_loader


def get_test_edges(data=arxiv_dataset(), year=[2020]):
    test_ids = []
    test_labels = []
    for y in year:
        test_ids+=(np.where(data['graph'].node_year == y)[0]).tolist()
    
    row, col = np.array(data['graph'].edge_index)
    adj_list = {}
    adj_list_inv = {}
    for src, dst in zip(row, col):
        if src not in adj_list:
            adj_list[src] = []
        adj_list[src].append(dst)
        if dst not in adj_list_inv:
            adj_list_inv[dst] = []
        adj_list_inv[dst].append(src)
            
    test_edges = []
    for idx in test_ids:
        for dst in adj_list.get(idx, []):
            test_edges.append((idx, dst))
            test_labels.append(1)
        while True:
            rdm = np.random.randint(0, data['graph'].num_nodes)
            if rdm not in adj_list_inv.get(idx, []) and rdm != idx:
                test_edges.append((idx, rdm))
                test_labels.append(0)
                break
    
    return test_edges, test_labels


class ClassifierDataset(TensorDataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

if __name__ == '__main__':
    data = arxiv_dataset()
    titleabs = load_titleabs()
    print(titleabs[:10])
