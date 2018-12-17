import tensorflow as tf

from datasets.single_dataset import extract_sequences
from preprocessors.eth_preprocessor import EthPreprosessor


def build_dataset(data_dir, image_size, obs_len, pred_len):
    preprocessor = EthPreprosessor(data_dir, image_size)

    # load and preprocess
    frame_df = preprocessor.preprocess_frame_data()

    # extract
    all_sequences = extract_sequences(frame_df, obs_len + pred_len)

    obs_true_seqs, pred_true_seqs = [], []
    for seq in all_sequences:
        obs_true_seqs.append(tf.cast(seq[:obs_len], tf.float32))
        pred_true_seqs.append(tf.cast(seq[obs_len:], tf.float32))

    return obs_true_seqs, pred_true_seqs
