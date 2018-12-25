from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

from datasets.load_single_dataset import load_single_dataset
from social_lstm.my_social_model_tfe import SocialLSTM
from social_lstm.trainer import Trainer


def load_args():
    default_out_dir = Path(Path(__file__).parent, '../data/outputs').absolute()

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


def main():
    tf.enable_eager_execution()
    args = load_args()

    train_ds = load_single_dataset(args.train_data_dirs, args.obs_len,
                                   args.pred_len)
    test_ds = load_single_dataset(args.test_data_dirs, args.obs_len,
                                  args.pred_len)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)

    model = SocialLSTM(args.pred_len, args.cell_side, args.n_side_cells,
                       args.lstm_dim, args.emb_dim)

    trainer = Trainer(model, optimizer, train_ds.take(6), test_ds.take(6),
                      args.n_epochs, args.out_dir)
    trainer.run()


if __name__ == '__main__':
    main()
