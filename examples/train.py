import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

sys.path.append((Path(__file__).parent / '..').absolute().as_posix())  # NOQA

from social_lstm_tf.datasets import load_single_dataset
from social_lstm_tf.social_lstm import metrics
from social_lstm_tf.social_lstm.losses import compute_loss
from social_lstm_tf.social_lstm.social_lstm import SocialLSTM


def load_args():
    parser = ArgumentParser()
    # train params
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--out_dir', type=str, required=True)

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

    # first save a configuration.
    _save_args_file(args, args.out_dir)

    model = SocialLSTM(args.pred_len, args.cell_side, args.n_side_cells,
                       args.lstm_dim, args.emb_dim)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)

    # prepare datasets
    train_ds, n_train_samples = load_single_dataset(
        args.train_data_dirs, args.obs_len, args.pred_len)
    test_ds, n_test_samples = load_single_dataset(
        args.test_data_dirs, args.obs_len, args.pred_len)

    # callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(
        Path(args.out_dir, 'logs').as_posix())

    def save_weights_func(epoch, _):
        model.save_weights(Path(args.out_dir, f'{epoch + 1:02d}.h5').as_posix())

    save_weights_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=save_weights_func)

    model.compile(optimizer=optimizer, loss=compute_loss,
                  metrics=[metrics.abe, metrics.fde])
    model.fit(x=train_ds, epochs=args.n_epochs, steps_per_epoch=n_train_samples,
              validation_data=test_ds, validation_steps=n_test_samples,
              callbacks=[tensorboard, save_weights_callback])


if __name__ == '__main__':
    main()
