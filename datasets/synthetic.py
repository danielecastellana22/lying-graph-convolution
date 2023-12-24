import os
import os.path as osp
from abc import ABC

import torch as th
import torch_geometric
from torch_geometric.datasets.fake import get_edge_index
from typing import List, Union, Tuple
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils as geom_utils


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
    sigma = th.ones((num_classes, num_node_features)) * 0.5
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

    edges_list = []
    for u in range(num_nodes):
        n_neighbours = max(int(th.ceil(avg_degree + 2 * th.randn(1)).item()), 1)

        if connection_type == 'random':
            possible_neigh = th.where(y != y[u])[0]
        elif connection_type == 'easy':
            target_cluster = (num_classes - 1 - y[u])
            if target_cluster == y[u]:
                # num_classes is odd
                target_cluster = 0
            possible_neigh = th.where(y == target_cluster)[0]
        else:
            raise ValueError(f'Connection type {connection_type} is not known!')

        aux = th.randperm(len(possible_neigh))
        for v in possible_neigh[aux[:n_neighbours]]:
            edges_list.append(th.tensor([u, v]))
            edges_list.append(th.tensor([v, u]))

    edge_index = th.stack(edges_list, dim=1)

    x, _ = __sample_node_features__(num_nodes, num_classes, num_node_features, y, node_features_overlap)

    data = Data(x=x, y=y, edge_index=geom_utils.coalesce(edge_index))
    return data


class Synthetic(InMemoryDataset, ABC):
    GENERATOR_FUN = {'multipartite': multipartite_graph_generator}
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

    def create_splits(self, n_splits=10, split_perc=None):

        if split_perc is None:
            split_perc = [70, 10, 20]

        # create the splits
        N = self._data.y.shape[0]
        self._data.train_mask = th.zeros((N, n_splits), dtype=th.bool)
        self._data.val_mask = th.zeros((N, n_splits), dtype=th.bool)
        self._data.test_mask = th.zeros((N, n_splits), dtype=th.bool)
        for i in range(n_splits):
            N_tr = (split_perc[0] * N) // 100
            N_val = (split_perc[1] * N) // 100
            N_test = N - N_tr - N_val  # (split_perc[2] * N) // 100
            aux = th.randperm(N)
            self._data.train_mask[aux[:N_tr], i] = True
            self._data.val_mask[aux[N_tr:N_tr + N_val], i] = True
            self._data.test_mask[aux[N_test:], i] = True
