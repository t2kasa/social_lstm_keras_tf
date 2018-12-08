from argparse import ArgumentParser
from pathlib import Path

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


def _load_single_dataset(data_dir, args):
    from datasets.build_dataset import build_dataset

    dataset_name = Path(data_dir).stem
    if dataset_name == 'hotel':
        image_size = (750, 576)
    else:  # eth
        image_size = (640, 480)

    obs_true_seqs, pred_true_seqs = build_dataset(data_dir, image_size,
                                                  args.obs_len, args.pred_len)
    return obs_true_seqs, pred_true_seqs


def main():
    tf.enable_eager_execution()
    args = load_args()

    # train_datasets = [_load_single_dataset(d, args) for d in
    #                   args.train_data_dirs]
    # test_datasets = [_load_single_dataset(d, args) for d in args.test_data_dirs]

    # xs, ys = train_datasets[0]
    # test_ds = test_datasets[0]
    # print(tf.shape(xs[0]))
    # exit(0)

    xs = [tf.random.normal([7, 4, 2])]
    social_lstm = SocialLSTM(args.pred_len, args.cell_side, args.n_side_cells,
                             args.lstm_dim, args.emb_dim)
    social_lstm.call(xs)


if __name__ == '__main__':
    main()
