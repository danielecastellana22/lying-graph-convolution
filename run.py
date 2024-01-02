import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch as th
import argparse
from datasets.utils import get_dataset
from utils.execution import parallel_model_selection
from utils.configuration import Config
from utils.misc import create_datatime_dir, eprint, string2class


def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--results-dir', dest='results_dir', default='results')
    parser.add_argument('--data-dir', dest='data_dir', default='data')
    parser.add_argument('--exp-config-file', dest='exp_config_file')
    parser.add_argument('--dataset-config-file', dest='dataset_config_file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--resume-dir', dest='resume_dir', default=None)
    parser.add_argument('--num-workers', dest='num_workers', default=10, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        th.set_anomaly_enabled(True)

    if args.resume_dir is not None:
        eprint('Resuming experiments! results-dir and config files will be ignored!')
        base_dir = args.resume_dir
        exp_config = Config.from_yaml_file(os.path.join(base_dir, 'grid.yaml'))
        ds = get_dataset(exp_config.storage_dir, exp_config.dataset_config)
    else:
        # read the config dict
        exp_config = Config.from_yaml_file(args.exp_config_file)
        dataset_config = Config.from_yaml_file(args.dataset_config_file)
        # create base directory for the experiment
        base_dir = create_datatime_dir(args.results_dir)

        # load the dataset just to start the download if needed
        ds = get_dataset(args.data_dir, dataset_config)
        exp_config['storage_dir'] = args.data_dir
        exp_config['dataset_config'] = dataset_config

    # select the training function according to the model class
    train_fun = string2class(exp_config.model_config['class']).get_training_fun()

    parallel_model_selection(
        train_config_fun=train_fun,
        exp_config=exp_config,
        n_splits=ds.n_splits,
        base_dir=base_dir,
        resume=args.resume_dir is not None,
        max_num_process=args.num_workers if not args.debug else 1)
