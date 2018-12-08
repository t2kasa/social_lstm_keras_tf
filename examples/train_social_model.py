import os
from argparse import Namespace, ArgumentParser
from shutil import copyfile

import matplotlib.pyplot as plt

from datasets.data_utils import obs_pred_split
from general_utils import dump_json_file
from general_utils import now_to_str
from load_model_config import ModelConfig
from load_model_config import load_model_config
from social_lstm.my_social_model import MySocialModel
from provide_train_test import provide_train_test


def load_train_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="data/results")
    return parser.parse_args()


def _make_weights_file_name(n_epochs: int) -> str:
    return "social_train_model_e{0:04d}.h5".format(n_epochs)


def train_social_model(out_dir: str, config: ModelConfig) -> None:
    # load data
    train_data, test_data = provide_train_test(config)

    # prepare train data
    obs_len_train, pred_len_train = obs_pred_split(config.obs_len,
                                                   config.pred_len,
                                                   *train_data)
    x_obs_len_train, _, grid_obs_len_train, zeros_obs_len_train = obs_len_train
    _, y_pred_len_train, _, _ = pred_len_train

    # prepare test data
    obs_len_test, pred_len_test = obs_pred_split(config.obs_len,
                                                 config.pred_len,
                                                 *test_data)
    x_obs_len_test, _, grid_obs_len_test, zeros_obs_len_test = obs_len_test
    _, y_pred_len_test, _, _ = pred_len_test

    os.makedirs(out_dir, exist_ok=True)

    # training
    my_model = MySocialModel(config)
    history = my_model.train_model.fit(
        [x_obs_len_train, grid_obs_len_train, zeros_obs_len_train],
        y_pred_len_train,
        batch_size=config.batch_size,
        epochs=config.n_epochs,
        verbose=1,
        validation_data=(
            [x_obs_len_test, grid_obs_len_test, zeros_obs_len_test],
            y_pred_len_test
        )
    )

    # save loss plot
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("social model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.savefig(os.path.join(out_dir, "test={}_loss.png".format(
        config.test_dataset_kind)))

    history_file = os.path.join(out_dir, "history.json")
    dump_json_file(history.history, history_file)

    # save the trained model weights
    weights_file = os.path.join(out_dir,
                                _make_weights_file_name(config.n_epochs))
    my_model.train_model.save_weights(weights_file)


def main():
    args = load_train_args()
    config = load_model_config(args.config)
    config.data_root = os.path.abspath(config.data_root)
    now_str = now_to_str()

    out_dir = os.path.join(args.out_root, "{}".format(now_str),
                           "test={}".format(config.test_dataset_kind))

    train_social_model(out_dir, config)
    copyfile(args.config,
             os.path.join(out_dir, os.path.basename(args.config)))


if __name__ == '__main__':
    main()
