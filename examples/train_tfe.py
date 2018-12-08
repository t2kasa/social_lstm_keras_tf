from argparse import ArgumentParser

import tensorflow as tf

from commons.general_utils import pxy_dim, out_dim
from social_lstm.my_social_model_tfe import SocialLSTM


def load_args():
    parser = ArgumentParser()
    # train params
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.003)

    # model params
    parser.add_argument('--obs_len', type=int, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--n_neighbor_pixels', type=int, required=True)
    parser.add_argument('--cell_side', type=float, required=True)
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
    xs = tf.random.normal([args.batch_size, args.obs_len, 52, pxy_dim])
    social_lstm = SocialLSTM(args.pred_len, args.cell_side, args.n_side_cells,
                             args.lstm_dim, args.emb_dim, out_dim)
    social_lstm.call(xs)


if __name__ == '__main__':
    main()
