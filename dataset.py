import pandas as pd
import os
import numpy as np
import random
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
import torch
import dgl
from scipy.sparse import coo_matrix


def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1

    return mask


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map

def load_pokec(dataset,sens_attr="region",predict_attr="I_am_working_in_field", path="dataset/pokec/", label_number=3000,sens_number=500,seed=20,test_idx=True):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    
    
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    # num_classes = len(idx_features_labels[predict_attr].unique()) - 1
    # labels = torch.eye(num_classes)[labels]
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    # import random
    # random.seed(seed)
    # label_idx = np.where(labels>=0)[0]
    # random.shuffle(label_idx)

    # idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    # idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    # if test_idx:
    #     idx_test = label_idx[label_number:]
    #     idx_val = idx_test
    # else:
    #     idx_test = label_idx[int(0.75 * len(label_idx)):]

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels > 0)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(
        label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(
        0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])


    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))

    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    # pokec data division
    labels[labels>1]=1
    if sens_attr:
        sens[sens>0]=1
        
    from collections import Counter
    print('predict_attr:',Counter(idx_features_labels[predict_attr]))
    print('sens_attr:',Counter(idx_features_labels[sens_attr]))
    print('total dimension:', features.shape)
    # random.shuffle(sens_idx)

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, adj

def load_NIFA(dataset):
    """Dataloader using graph data preprocessed by NIFA"""
    import os
    import dgl
    from scipy.sparse import coo_matrix
    
    # Check if a graph file exists
    graph_path = f'./output/{dataset}.bin'
    if os.path.exists(graph_path):
        print(f"Loading graph for dataset: {dataset} from {graph_path}")
        
        # Load graph
        glist, _ = dgl.load_graphs(graph_path)
        g = glist[0]

        # Extract attributes from the graph
        idx_train = torch.where(g.ndata['train_index'])[0]
        idx_val = torch.where(g.ndata['val_index'])[0]
        idx_test = torch.where(g.ndata['test_index'])[0]

        features = g.ndata['feature']
        labels = g.ndata['label']
        sens = g.ndata['sensitive']
        print(g.ndata)


        adj = coo_matrix(
            (np.ones(g.edges()[0].shape[0]), (g.edges()[0].numpy(), g.edges()[1].numpy())),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32
        )
        idx_sens_train = idx_train
        
        adj_norm = sys_normalized_adjacency(adj)
        adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

        edge_index, _ = from_scipy_sparse_matrix(adj)

        train_mask = index_to_mask(features.shape[0], idx_train)
        val_mask = index_to_mask(features.shape[0], idx_val)
        test_mask = index_to_mask(features.shape[0], idx_test)

        print("Graph successfully loaded.")
        return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, adj
    elif not os.path.exists(graph_path):
        raise FileNotFoundError(f'The provided path {graph_path} does not exist')


def get_dataset(dataname):
    # todo: check sens_idx of pokec
    if(dataname == 'pokec_z_poisoned' or dataname == 'pokec_n' or dataname == 'pokec_z' or dataname == 'pokec_n_poisoned'):
        sens_idx = 3
    # todo: check sens_idx of dblp
    elif(dataname == 'dblp' or dataname == 'dblp_poisoned'):
        sens_idx = 3
    else:
        raise ValueError(f"Invalid dataset name: '{dataname}'. The dataset name must be one of the valid options: "
                     "'pokec_z_poisoned', 'pokec_n', 'pokec_z', 'pokec_n_poisoned', 'dblp', 'dblp_poisoned'.")
    
    adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, adj = load_NIFA(dataname)

    x_max, x_min = torch.max(features, dim=0)[
        0], torch.min(features, dim=0)[0]
    
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    features = norm_features

    return Data(adj=adj, x=features, edge_index=edge_index, adj_norm_sp=adj_norm_sp, y=labels.float(), train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, sens=sens), sens_idx, x_min, x_max
