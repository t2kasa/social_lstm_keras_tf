from pathlib import Path

import tensorflow as tf

from datasets.single_dataset import extract_sequences
from preprocessors.preprocess_ewap import EwapPreprocessor


def build_dataset(data_dir, image_size, obs_len, pred_len):
    preprocessor = EwapPreprocessor(data_dir, image_size)

    # load and preprocess
    frame_df = preprocessor.preprocess_frame_data()

    # extract
    all_sequences = extract_sequences(frame_df, obs_len + pred_len)

    obs_true_seqs, pred_true_seqs = [], []
    for seq in all_sequences:
        obs_true_seqs.append(tf.cast(seq[:obs_len], tf.float32))
        pred_true_seqs.append(tf.cast(seq[obs_len:], tf.float32))

    return obs_true_seqs, pred_true_seqs


def load_single_dataset(data_dir, args):
    dataset_name = Path(data_dir).stem
    if dataset_name == 'hotel':
        image_size = (750, 576)
    else:  # eth
        image_size = (640, 480)

    obs_true_seqs, pred_true_seqs = build_dataset(data_dir, image_size,
                                                  args.obs_len, args.pred_len)

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
