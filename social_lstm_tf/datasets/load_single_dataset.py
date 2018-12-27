from functools import reduce

import numpy as np
import tensorflow as tf

from social_lstm_tf.preprocessors.preprocess_data import preprocess_data


def load_single_dataset(data_dirs, obs_len, pred_len, shuffle=True,
                        batch_size=1):
    """Builds a single dataset from one or more data directories.

    :param str|list[str] data_dirs: one or more data directories.
    :param int obs_len: observation sequence length.
    :param int pred_len: prediction sequence length.
    :return: a single dataset and the number of samples.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    seqs = [build_obs_pred_sequences(d, obs_len, pred_len) for d in data_dirs]
    obs_seqs, pred_seqs = zip(*seqs)
    obs_seqs, pred_seqs = sum(obs_seqs, []), sum(pred_seqs, [])

    n_samples = len(obs_seqs)
    obs_ds = tf.data.Dataset.from_generator(
        _seqs_generator(obs_seqs), tf.float32,
        tf.TensorShape([obs_len, None, 2]))
    pred_ds = tf.data.Dataset.from_generator(
        _seqs_generator(pred_seqs), tf.float32,
        tf.TensorShape([pred_len, None, 2]))

    ds = tf.data.Dataset.zip((obs_ds, pred_ds))
    if shuffle:
        ds = ds.shuffle(n_samples)
    ds = ds.batch(batch_size).repeat()
    return ds, n_samples


def build_obs_pred_sequences(data_dir, obs_len, pred_len):
    pos_df = preprocess_data(data_dir)
    all_sequences = extract_sequences(pos_df, obs_len + pred_len)

    obs_true_seqs, pred_true_seqs = [], []
    for seq in all_sequences:
        obs_true_seqs.append(tf.cast(seq[:obs_len], tf.float32))
        pred_true_seqs.append(tf.cast(seq[obs_len:], tf.float32))

    return obs_true_seqs, pred_true_seqs


def extract_sequences(frame_df, seq_len):
    """Extracts sequences as a dataset.

    :param frame_df: tabled pedestrian positions data. it is expected that the
        data frame has four columns 'frame', 'id', 'x', and 'y'.
    :param seq_len: each sequence length.
    :return: [t, t + seq_len) sequences.
    """
    sequences = []
    all_frames = frame_df['frame'].unique()
    for i in range(len(all_frames) - seq_len + 1):
        frame_range = all_frames[i:i + seq_len]
        df = frame_df[frame_df['frame'].isin(frame_range)]

        # collect pedestrian ids when the pedestrians exist in the all frames
        target_pids = _extract_pids_in_all_frames(df)
        # skip when there are no pedestrians
        if not target_pids:
            continue

        curr_target_df = df[df['id'].isin(target_pids)]
        # built sequence shape is (seq_len, n_pids, 2)
        curr_seq = _build_sequence(curr_target_df)
        sequences.append(curr_seq)

    return sequences


def _seqs_generator(seqs):
    def gen():
        for x in seqs:
            yield x

    return gen


def _extract_pids_in_all_frames(df):
    pids = set(
        reduce(np.intersect1d, [g['id'] for _, g in df.groupby('frame')]))
    return pids


def _build_sequence(target_df):
    seq = np.array([np.array(df[['x', 'y']]) for _, df in
                    target_df.groupby('frame')])
    return seq
