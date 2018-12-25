import json
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

from datasets.load_single_dataset import load_single_dataset
from social_lstm.my_social_model_tfe import SocialLSTM
from social_lstm.trainer import Trainer


def load_args():
    default_out_dir = Path(Path(__file__).parent,
                           '../data/outputs').absolute().as_posix()

    parser = ArgumentParser()
    # train params
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--out_dir', type=str, default=default_out_dir)

    # model params
    parser.add_argument('--obs_len', type=int, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--n_neighbor_pixels', type=int, default=32)
    parser.add_argument('--cell_side', type=float, default=0.5)
    parser.add_argument('--n_side_cells', type=int, default=8)
    parser.add_argument('--lstm_dim', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=64)

    # dataset config
    parser.add_argument('--train_data_dirs', type=str, nargs='+')
    parser.add_argument('--test_data_dirs', type=str, nargs='+')

    args, _ = parser.parse_known_args()
    return args


def _save_args_file(args, out_dir, out_file_name='train_config.json'):
    with open(Path(out_dir, out_file_name), 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)


def main():
    tf.enable_eager_execution()
    args = load_args()

    model = SocialLSTM(args.pred_len, args.cell_side, args.n_side_cells,
                       args.lstm_dim, args.emb_dim)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)

    train_ds = load_single_dataset(
        args.train_data_dirs, args.obs_len, args.pred_len)
    test_ds = load_single_dataset(
        args.test_data_dirs, args.obs_len, args.pred_len)

    trainer = Trainer(model, optimizer, train_ds.take(1), test_ds.take(1),
                      args.n_epochs, args.out_dir)
    trainer.run()

    _save_args_file(args, args.out_dir)


if __name__ == '__main__':
    main()
