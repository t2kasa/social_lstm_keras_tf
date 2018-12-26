import tensorflow as tf

from social_lstm_tf.social_lstm.tfe_normal_sampler import normal2d_sample


def abe(pos_future_true, o_future_pred):
    """Computes Average Displacement Error (ABE).

    :param o_future_pred: (batch_size, seq_len, n_pedestrians, 5)
    :param pos_future_true: (batch_size, seq_len, n_pedestrians, 2)
    :return: the abe.
    """
    pos_future_true = tf.reshape(pos_future_true, [-1, 2])
    o_future_pred = tf.reshape(o_future_pred, [-1, 5])

    pos_future_pred = normal2d_sample(o_future_pred)
    return tf.reduce_mean(tf.square(pos_future_true - pos_future_pred))


def fde(pos_future_true, o_future_pred):
    """Computes Final Displacement Error (FBE).

    :param o_future_pred: (batch_size, seq_len, n_pedestrians, 5)
    :param pos_future_true: (batch_size, seq_len, n_pedestrians, 2)
    :return: the fde.
    """
    pos_future_true = tf.reshape(pos_future_true[:, -1, :, :], [-1, 2])
    o_future_pred = tf.reshape(o_future_pred[:, -1, :, :], [-1, 5])

    pos_future_pred = normal2d_sample(o_future_pred)
    return tf.reduce_mean(tf.square(pos_future_true - pos_future_pred))
