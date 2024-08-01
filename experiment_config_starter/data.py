import os

from typing import Tuple, Optional
from datasets import load_from_disk, Dataset
from config import ExperimentConfig


def load_dataset(experiment_cfg: ExperimentConfig) -> Tuple[Dataset, Optional[Dataset]]:
    data_dir = experiment_cfg.data_dir
    train_file = experiment_cfg.train_file
    eval_dir = experiment_cfg.eval_dir
    train_size = experiment_cfg.train_size

    train_dataset = load_from_disk(os.path.join(data_dir, train_file))

    if eval_dir is not None:
        eval_dataset = load_from_disk(eval_dir)
    else:
        eval_dataset = None

    if train_size is not None:
        raw_train = raw_train["train"].take(train_size)

    return train_dataset, eval_dataset
