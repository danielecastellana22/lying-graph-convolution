import os
import os.path as osp
from abc import ABC

import torch as th
import torch_geometric
from torch_geometric.datasets.fake import get_edge_index
from typing import List, Union, Tuple
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils as geom_utils
from sklearn.model_selection import train_test_split
import numpy as np


def __build_torch_geometric_fake_graph__(num_nodes, avg_degree):
    ok = False
    edge_index = None
    # to ensure connected graph
    while not ok:
        edge_index = get_edge_index(num_nodes, num_nodes, avg_degree,
                                    is_undirected=True, remove_loops=True)
        ok = not th.any(torch_geometric.utils.degree(edge_index[0]) == 0).item()

    return edge_index


def __sample_node_features__(num_nodes, num_classes, num_node_features, cluster_id=None, node_features_overlap=0):
    sigma = th.ones((num_classes, num_node_features))
    mu = (10*0.5) * (1 - node_features_overlap) * th.arange(num_classes).view(-1, 1).expand(-1, num_node_features)

    if cluster_id is None:
        cluster_id = th.randint(0, num_classes, size=[num_nodes])
    x = mu[cluster_id] + sigma[cluster_id] * th.randn(size=(num_nodes, num_node_features))
    return x, cluster_id


def multipartite_graph_generator(num_nodes, avg_degree, num_classes, num_node_features, connection_type,
                                 node_features_overlap):

    # assign each node to a group
    num_nodes_for_class = num_nodes // num_classes + 1
    y = th.arange(0, num_nodes, dtype=th.long) // num_nodes_for_class

    n_neighbours_list = th.clamp(th.ceil(avg_degree + 2 * th.randn(num_nodes)), min=1).to(int)
    edges_list = []
    for u in range(num_nodes):

        if connection_type == 'random':
            possible_neigh_mask = y != y[u]
        elif connection_type == 'easy':
            target_cluster = (num_classes - 1 - y[u])
            if target_cluster == y[u]:
                # num_classes is odd
                target_cluster = 0
            possible_neigh_mask = y == target_cluster
        else:
            raise ValueError(f'Connection type {connection_type} is not known!')

        possible_neigh_mask = th.logical_and(possible_neigh_mask, n_neighbours_list > 0)
        possible_neigh = th.where(possible_neigh_mask)[0]
        n_neighbours = min(n_neighbours_list[u], len(possible_neigh))
        if n_neighbours > 0:
            aux = th.randperm(len(possible_neigh))
            for v in possible_neigh[aux[:n_neighbours]]:
                edges_list.append(th.tensor([u, v]))
                edges_list.append(th.tensor([v, u]))
                n_neighbours_list[v] -= 1

        n_neighbours_list[u] = 0

    edge_index = th.stack(edges_list, dim=1)

    x, _ = __sample_node_features__(num_nodes, num_classes, num_node_features, y, node_features_overlap)

    data = Data(x=x, y=y, edge_index=geom_utils.coalesce(edge_index))
    return data


def community_graph_generator(num_nodes, avg_degree, num_classes, num_node_features, p_intra_community,
                              node_features_overlap):

    # assign each node to a group
    num_nodes_for_class = num_nodes // num_classes + 1
    y = th.arange(0, num_nodes, dtype=th.long) // num_nodes_for_class

    n_neighbours_list = th.clamp(th.ceil(avg_degree + 2 * th.randn(num_nodes)), min=1).to(int)
    n_neighbours_intra = th.tensor([th.sum(th.rand(n_neighbours_list[u]) < p_intra_community) for u in range(num_nodes)])
    n_neighbours_extra = n_neighbours_list - n_neighbours_intra
    assert th.all(n_neighbours_extra>=0) and th.all(n_neighbours_intra>=0)

    edges_list = []
    for u in range(num_nodes):

        possible_intra_neigh = th.where(th.logical_and(y == y[u], n_neighbours_list > 0))[0]
        possible_extra_neigh = th.where(th.logical_and(y != y[u], n_neighbours_list > 0))[0]

        for type, n_neighbours, possible_neigh in [(0, n_neighbours_intra[u], possible_intra_neigh),
                                                   (1, n_neighbours_extra[u], possible_extra_neigh)]:
            n_neighbours = min(n_neighbours, len(possible_neigh))
            if n_neighbours > 0:
                aux = th.randperm(len(possible_neigh))
                for v in possible_neigh[aux[:n_neighbours]]:
                    edges_list.append(th.tensor([u, v]))
                    edges_list.append(th.tensor([v, u]))
                    n_neighbours_list[v] -= 1
                    if type == 0:
                        # intra neighbour
                        n_neighbours_intra[v] -= 1
                    else:
                        # extra neighbour
                        n_neighbours_extra[v] -= 1

        n_neighbours_list[u] = 0

    edge_index = th.stack(edges_list, dim=1)

    x, _ = __sample_node_features__(num_nodes, num_classes, num_node_features, y, node_features_overlap)

    data = Data(x=x, y=y, edge_index=geom_utils.coalesce(edge_index))
    return data


class Synthetic(InMemoryDataset, ABC):
    GENERATOR_FUN = {'multipartite': multipartite_graph_generator,
                     'sbm': community_graph_generator}
                     #'count-neighbours-type': count_neighbours_type_graph_generator,
                     #'count-triangles': count_triangles_graph_generator}

    POSSIBLE_NAMES = list(GENERATOR_FUN.keys())

    def __init__(self, root: str, name: str, num_nodes, avg_degree=None, num_classes=None, num_node_features=1,
                 **other_params):
        super().__init__(root)
        if name not in self.POSSIBLE_NAMES:
            raise ValueError(f'Dataset name {name} is not known!')

        self._name = name
        self._num_nodes = num_nodes
        self._avg_degree = avg_degree
        self._num_node_features = num_node_features
        self._num_classes = num_classes
        self._other_name = ''
        for k,v in other_params.items():
            self._other_name += str(v) + '_'
        self._other_name = self._other_name[:-1]

        if osp.exists(self.processed_paths[0]):
            print('Loading the stored data...')
            self._data, self.slices = th.load(self.processed_paths[0])
        else:
            # we generate the data
            print('Generating the new data...')
            data = self.GENERATOR_FUN[name](num_nodes, avg_degree, num_classes, num_node_features, **other_params)
            self._data, self.slices = self.collate([data])
            self.create_splits()
            os.makedirs(self.processed_dir)
            th.save((self._data, self.slices), self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        raise ValueError('Synthetic datases have no raw dir')
        # return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self._name, self._other_name, f'{self._num_nodes}_{self._num_classes}_{self._avg_degree}', 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        raise ValueError('Synthetic datases have no raw files')

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    def create_splits(self, n_splits=10, test_fraction=0.2, val_fraction=0.2):

        # create the splits
        y = self._data.y.numpy()
        N = y.shape[0]
        N_test = int(N*test_fraction)
        N_val = int(N*val_fraction)


        self._data.train_mask = th.zeros((N, n_splits), dtype=th.bool)
        self._data.val_mask = th.zeros((N, n_splits), dtype=th.bool)
        self._data.test_mask = th.zeros((N, n_splits), dtype=th.bool)
        for i in range(n_splits):
            train_val_idx, test_idx = train_test_split(np.arange(N), test_size=N_test, stratify=y)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=N_val, stratify=y[train_val_idx])

            self._data.train_mask[train_idx, i] = True
            self._data.val_mask[val_idx, i] = True
            self._data.test_mask[test_idx, i] = True
