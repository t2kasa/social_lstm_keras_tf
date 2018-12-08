import numpy as np

from commons.general_utils import DatasetKind
from commons.general_utils import _check_dataset_kind
from commons.general_utils import get_data_dir
from datasets.load_dataset import load_dataset_from_config, load_dataset
from commons.load_model_config import ModelConfig


def provide_train_test(config: ModelConfig):
    all_dataset_kinds = set(DatasetKind)

    test_dataset_kind = _check_dataset_kind(config.test_dataset_kind)
    train_dataset_kinds = all_dataset_kinds - {test_dataset_kind}

    # load (several) train datasets
    x_train, y_train, grid_train, zeros_train = [], [], [], []
    for train_dataset_kind in train_dataset_kinds:
        data_dir = get_data_dir(config.data_root, train_dataset_kind)

        train_dataset = load_dataset(data_dir, train_dataset_kind,
                                     config.obs_len + config.pred_len,
                                     config.max_n_peds,
                                     config.n_neighbor_pixels, config.grid_side)

        x, y, g, z = train_dataset.get_data(config.lstm_state_dim)
        x_train.append(x)
        y_train.append(y)
        grid_train.append(g)
        zeros_train.append(z)

    x_train = np.concatenate(x_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.float32)
    grid_train = np.concatenate(grid_train, axis=0).astype(np.float32)
    zeros_train = np.concatenate(zeros_train, axis=0).astype(np.float32)

    train_data = (x_train, y_train, grid_train, zeros_train)

    # load (one) test dataset
    test_dataset = load_dataset_from_config(config)
    x_test, y_test, grid_test, zeros_test = test_dataset.get_data(
        config.lstm_state_dim)
    test_data = (x_test, y_test, grid_test, zeros_test)

    return train_data, test_data
