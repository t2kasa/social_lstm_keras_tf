import tensorflow as tf

from social_lstm.tfe_normal_sampler import normal2d_log_pdf


def compute_loss(o_pred, pos_true):
    """Computes negative log likelihood loss.

    :param o_pred: (..., 5)
    :param pos_true: (..., 2)
    :return: the computed loss.
    """
    o_pred = tf.reshape(o_pred, [-1, 5])
    pos_true = tf.reshape(pos_true, [-1, 2])

    log_probs = normal2d_log_pdf(o_pred, pos_true)
    loss_value = -tf.reduce_sum(log_probs)
    return loss_value
