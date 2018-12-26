import tensorflow as tf

from social_lstm_tf.social_lstm.tfe_normal_sampler import normal2d_log_pdf


def compute_loss(pos_future_true, o_future_pred):
    """Computes negative log likelihood loss.

    :param o_future_pred: (..., 5)
    :param pos_future_true: (..., 2)
    :return: the computed loss.
    """
    o_future_pred = tf.reshape(o_future_pred, [-1, 5])
    pos_future_true = tf.reshape(pos_future_true, [-1, 2])

    log_probs = normal2d_log_pdf(o_future_pred, pos_future_true)
    loss_value = -tf.reduce_sum(log_probs)
    return loss_value
