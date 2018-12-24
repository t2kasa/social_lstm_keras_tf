from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

from datasets.load_single_dataset import load_single_dataset
from social_lstm.losses import compute_loss
from social_lstm.metrics import compute_abe_tf
from social_lstm.my_social_model_tfe import SocialLSTM
from social_lstm.tfe_normal_sampler import normal2d_sample


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


def train_epoch(social_lstm, optimizer, train_ds, step_counter, log_interval=5):
    for batch_index, (pos_true_obs, pos_true_pred) in enumerate(train_ds):
        with tf.contrib.summary.record_summaries_every_n_global_steps(
                log_interval, global_step=step_counter):
            with tf.GradientTape() as tape:
                o_pred = social_lstm(pos_true_obs)
                loss_value = compute_loss(o_pred, pos_true_pred)

            pos_true_pred = tf.reshape(pos_true_pred, [-1, 2])
            pos_pred_pred = normal2d_sample(o_pred)
            abe = compute_abe_tf(pos_true_pred, pos_pred_pred)

            tf.contrib.summary.scalar('abe', abe)
            tf.contrib.summary.scalar('loss', loss_value)
            # compute grads and update model params
            grads = tape.gradient(loss_value, social_lstm.variables)
            optimizer.apply_gradients(zip(grads, social_lstm.variables),
                                      global_step=step_counter)

            if step_counter.numpy() % log_interval == 0:
                print(f'step: #{batch_index + 1:06d}\tloss: {loss_value}\t'
                      f'abe: {abe}')


def eval_epoch(social_lstm, test_ds):
    loss_mean = tf.contrib.eager.metrics.Mean('loss', tf.float32)
    abe_mean = tf.contrib.eager.metrics.Mean('abe', tf.float32)

    for pos_true_obs, pos_true_pred in test_ds:
        o_pred = social_lstm(pos_true_obs)
        loss_mean(compute_loss(o_pred, pos_true_pred))

        pos_true_pred = tf.reshape(pos_true_pred, [-1, 2])
        pos_pred_pred = normal2d_sample(o_pred)
        abe_mean(compute_abe_tf(pos_true_pred, pos_pred_pred))

    print(f'test loss: {loss_mean.result()}, abe: {abe_mean.result()}')
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', loss_mean.result())
        tf.contrib.summary.scalar('abe', abe_mean.result())


def main():
    tf.enable_eager_execution()
    args = load_args()

    train_ds = load_single_dataset(args.train_data_dirs, args.obs_len,
                                   args.pred_len)
    test_ds = load_single_dataset(args.test_data_dirs, args.obs_len,
                                  args.pred_len)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)

    train_dir = str(Path(args.out_dir, 'train'))
    test_dir = str(Path(args.out_dir, 'test'))
    tf.gfile.MakeDirs(args.out_dir)

    train_summary_writer = tf.contrib.summary.create_file_writer(
        train_dir, name='train')
    test_summary_writer = tf.contrib.summary.create_file_writer(
        test_dir, name='test')

    step_counter = tf.train.get_or_create_global_step()

    social_lstm = SocialLSTM(args.pred_len, args.cell_side, args.n_side_cells,
                             args.lstm_dim, args.emb_dim)

    for _ in range(args.n_epochs):
        with train_summary_writer.as_default():
            train_epoch(social_lstm, optimizer, train_ds.take(11), step_counter)
        with test_summary_writer.as_default():
            eval_epoch(social_lstm, test_ds.take(11))


if __name__ == '__main__':
    main()
