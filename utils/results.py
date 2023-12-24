import os
import os.path as osp
import numpy as np
from .configuration import Config
from .serialisation import from_json_file, from_torch_file


class ModelSelectionResult:

    def __init__(self, exp_dir_path):

        self.exp_dir_path = exp_dir_path
        self.exp_config = Config.from_yaml_file(os.path.join(exp_dir_path, 'grid.yaml'))
        self.grid = self.exp_config.get_grid()
        self.grid_shape = tuple([len(v) for k,v in self.grid.items()])
        self.count_exp_finished = 0
        self.tot_exp = 0
        self.dict_results = {}
        self.update()

    def update(self):
        self.count_exp_finished = 0

        # compute the number of config dir
        list_config_dir_name = [x for x in os.listdir(self.exp_dir_path) if x.startswith('config')]
        n_configs = len(list_config_dir_name)
        self.n_configs = n_configs

        # compute the number of split dir
        list_split_dir_name = [x for x in os.listdir(osp.join(self.exp_dir_path, 'config_0')) if x.startswith('split')]
        n_splits = len(list_split_dir_name)
        self.n_splits = n_splits

        self.dict_results = {}
        self.tot_exp = n_configs * n_splits

        for config_dir_name in list_config_dir_name:
            config_idx = int(config_dir_name.split('_')[-1])
            for split_dir_name in list_split_dir_name:
                split_idx = int(split_dir_name.split('_')[-1])
                split_dir_path = osp.join(self.exp_dir_path, config_dir_name, split_dir_name)

                results_file = osp.join(split_dir_path, 'training_results.json')
                if osp.exists(results_file):
                    self.count_exp_finished += 1
                    current_res_d = from_json_file(results_file)
                    for k in current_res_d:
                        if k not in self.dict_results:
                            self.dict_results[k] = np.full((n_configs, n_splits), fill_value=np.nan, dtype=object)

                        self.dict_results[k][config_idx, split_idx] = current_res_d[k]

    def __getitem__(self, key_list):
        metric = key_list[0]
        split_idx = key_list[-1]
        config_idx = np.ravel_multi_index(key_list[1:-1], self.grid_shape)
        if metric in self.dict_results:
            return self.dict_results[metric][config_idx, split_idx]
        else:
            # it is a file to load
            file_path = osp.join(self.exp_dir_path, f'config_{config_idx}', f'split_{split_idx}', f'{metric}.pt')
            return from_torch_file(file_path)

    def get_test_results(self):
        best_config_idx = np.nanargmax(self.dict_results['validation_metric'], axis=0)
        return self.dict_results['test_metric'][best_config_idx, np.arange(self.n_splits)], best_config_idx

    def get_avg_val_results(self, k_list):
        avg_res = self.dict_results['validation_metric'].mean(-1).reshape(self.grid_shape)
        labels = []
        for i, k in enumerate(self.grid):
            if k not in k_list:
                avg_res = np.max(avg_res, i, keepdims=True)
            else:
                labels.append(self.grid[k])

        return avg_res.squeeze() #, np.array(labels).squeeze()

