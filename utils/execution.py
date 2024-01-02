import os
import os.path as osp
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .serialisation import from_json_file
from .configuration import Config
from .misc import eprint, set_initial_seed


def parallel_model_selection(train_config_fun,
                             exp_config: Config,
                             n_splits: int,
                             base_dir: str,
                             resume: bool,
                             max_num_process: int):
    # save the grid search
    exp_config.to_yaml_file(os.path.join(base_dir, 'grid.yaml'))
    config_list = exp_config.build_config_grid()
    n_configs = len(config_list)

    if max_num_process > 1 and n_configs > 1:
        process_pool = ProcessPoolExecutor(max_num_process)
    else:
        process_pool = None
        eprint('No process pool created! The execution will be sequential!')

    print(f'Model selection with {len(config_list)} configurations started!')

    for config_idx, config in enumerate(config_list):
        for split_idx in range(n_splits):
            exp_dir = os.path.join(base_dir, f'config_{config_idx}', f'split_{split_idx}')
            if resume and os.path.exists(osp.join(exp_dir,'training_results.json')):
                eprint(f'config_{config_idx} split_{split_idx} is already done!')
                continue

            os.makedirs(exp_dir, exist_ok=resume)

            params = {'config': config, 'split_idx': split_idx, 'exp_dir': exp_dir,
                      'write_on_console': process_pool is None}

            if process_pool is None:
                train_config_fun(**params)
            else:
                f = process_pool.submit(train_config_fun, **params)

    if process_pool is not None:
        process_pool.shutdown()

    print(f'Model selection terminated!')
