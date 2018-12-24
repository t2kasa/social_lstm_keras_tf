import tensorflow as tf

from datasets.single_dataset import extract_sequences
from preprocessors.preprocess_data import preprocess_data


def build_obs_pred_sequences(data_dir, obs_len, pred_len):
    pos_df = preprocess_data(data_dir)
    all_sequences = extract_sequences(pos_df, obs_len + pred_len)

    obs_true_seqs, pred_true_seqs = [], []
    for seq in all_sequences:
        obs_true_seqs.append(tf.cast(seq[:obs_len], tf.float32))
        pred_true_seqs.append(tf.cast(seq[obs_len:], tf.float32))

    return obs_true_seqs, pred_true_seqs


def load_single_dataset(data_dir, args):
    obs_true_seqs, pred_true_seqs = build_obs_pred_sequences(
        data_dir, args.obs_len, args.pred_len)

    n_seqs = len(obs_true_seqs)
    obs_ds = tf.data.Dataset.from_generator(
        _seqs_generator(obs_true_seqs), tf.float32,
        tf.TensorShape([args.obs_len, None, 2]))
    pred_ds = tf.data.Dataset.from_generator(
        _seqs_generator(pred_true_seqs), tf.float32,
        tf.TensorShape([args.pred_len, None, 2]))

    return tf.data.Dataset.zip((obs_ds, pred_ds)).shuffle(n_seqs).batch(1)


def _seqs_generator(seqs):
    def gen():
        for x in seqs:
            yield x

    return gen
