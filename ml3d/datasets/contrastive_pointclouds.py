import numpy as np
from os.path import join
from pathlib import Path
from glob import glob
import joblib
import logging
import random

from .base_dataset import BaseDataset
from ..utils import DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class ContrastivePointclouds(BaseDataset):
    """TODO
    """

    def __init__(self,
                 dataset_path,
                 name='ContrastivePointclouds',
                 hold_out_frac=0.2,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.
        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (ContrastivePointclouds in this case).
            hold_out_frac: fraction of data to be partitioned into hold out set.
        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path, name=name, **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path

        self.input_files = glob(join(cfg.dataset_path, 'input', '*.pkl'))
        self.input_files.sort()
        self.input_files = np.array(self.input_files)
        self.pair_files = glob(join(cfg.dataset_path, 'pair', '*.pkl'))
        self.pair_files.sort()
        self.pair_files = np.array(self.pair_files)

        assert (len(self.pair_files) == len(self.input_files))

        split_idx = int(hold_out_frac * len(self.input_files))
        self.holdout_input_files = self.input_files[:split_idx]
        self.holdout_pair_files = self.pair_files[:split_idx]
        self.input_files = self.input_files[split_idx:]
        self.pair_files = self.pair_files[split_idx:]

    @staticmethod
    def read_pc(path):
        """Reads lidar data from the path provided.
        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        return joblib.load(path)

    @staticmethod
    def get_label_to_names():
        return dict()

    def get_split(self, split):
        """Returns a dataset split.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return ContrastivePointcloudsSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        Returns:
            A dataset split object providing the requested subset of the data.
        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['train', 'training']:
            return self.input_files, self.pair_files
        elif split in ['test', 'testing', 'valid', 'validation']:
            return self.holdout_input_files, self.holdout_pair_files
        elif split in ['all']:
            return self.input_files + self.holdout_input_files, \
                   self.pair_files + self.holdout_pair_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.
        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.
        Returns:
            If the dataum attribute is tested, then resturn the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attrs):
        """Saves the output of a model.
        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            attrs: The attributes that correspond to the outputs passed in
            results.
        """
        pass


class ContrastivePointcloudsSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg
        input_files, pair_files = dataset.get_split_list(split)
        assert (len(input_files) == len(pair_files))
        log.info("Found {} pointclouds for {}".format(len(input_files), split))

        self.input_files = input_files
        self.pair_files = pair_files
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.input_files)

    def negative_idxs(self, idx, num_negatives=8, min_idx_distance=10):
        choices = len(self.input_files) - 2 * min_idx_distance
        assert choices >= num_negatives
        rng_state = np.random.get_state()
        np.random.seed(idx)
        negative_offsets = np.random.choice(choices, num_negatives)
        np.random.set_state(rng_state)

        # Offset from idx the min_idx distance, then use negative_offsets for variability,
        # with modulus for wrap around below max index.
        negitive_idxs = (negative_offsets +
                         (min_idx_distance + idx)) % len(self.input_files)
        return negitive_idxs

    def get_data(self, idx):
        input = self.dataset.read_pc(self.input_files[idx])
        pair = self.dataset.read_pc(self.pair_files[idx])

        negative_files = self.input_files[self.negative_idxs(idx)]
        negatives = [self.dataset.read_pc(f) for f in negative_files]
        data = {'input': input, 'pair': pair, 'negatives': negatives}
        return data

    def get_attr(self, idx):
        input_path = self.input_files[idx]
        name = Path(input_path).name.split('.')[0]

        attr = {'name': name, 'path': input_path, 'split': self.split}
        return attr


DATASET._register_module(ContrastivePointclouds)