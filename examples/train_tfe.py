from argparse import ArgumentParser

import tensorflow as tf

from datasets.load_single_dataset import load_single_dataset
from social_lstm.my_social_model_tfe import SocialLSTM


def load_args():
    parser = ArgumentParser()
    # train params
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.003)

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

    train_ds_list = [load_single_dataset(d, args) for d in args.train_data_dirs]
    # test_ds_list = [load_single_dataset(d, args) for d in args.test_data_dirs]

    # test_datasets = [_load_single_dataset(d, args) for d in args.test_data_dirs]

    x_ds, y_ds = train_ds_list[0]

    # test_ds = test_datasets[0]

    # xs = [tf.random.normal([7, 4, 2])]
    social_lstm = SocialLSTM(args.pred_len, args.cell_side, args.n_side_cells,
                             args.lstm_dim, args.emb_dim)

    for i, x in enumerate(x_ds):
        print(i)
        social_lstm.call(x)


if __name__ == '__main__':
    main()
