import json
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

from datasets.load_single_dataset import load_single_dataset
from social_lstm import metrics
from social_lstm.losses import compute_loss
from social_lstm.my_social_model_tfe import SocialLSTM


def load_args():
    parser = ArgumentParser()
    # train params
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--out_dir', type=str, default='../data/outputs')

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
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir, out_file_name).as_posix(), 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)


def main():
    tf.enable_eager_execution()
    args = load_args()

    model = SocialLSTM(args.pred_len, args.cell_side, args.n_side_cells,
                       args.lstm_dim, args.emb_dim)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)

    # prepare datasets
    train_ds, n_train_samples = load_single_dataset(
        args.train_data_dirs, args.obs_len, args.pred_len)
    test_ds, n_test_samples = load_single_dataset(
        args.test_data_dirs, args.obs_len, args.pred_len)

    tensorboard = tf.keras.callbacks.TensorBoard(
        Path(args.out_dir, 'logs').as_posix())
    model.compile(optimizer=optimizer, loss=compute_loss,
                  metrics=[metrics.abe, metrics.fde])
    model.fit(x=train_ds, steps_per_epoch=n_train_samples,
              validation_data=test_ds, validation_steps=n_test_samples,
              callbacks=[tensorboard])

    # save the trained model weights and configuration.
    model.save_weights(Path(args.out_dir, 'saved_model.h5').as_posix())
    _save_args_file(args, args.out_dir)


if __name__ == '__main__':
    main()
