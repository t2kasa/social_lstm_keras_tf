import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.contrib.keras.api.keras.layers import Multiply


def build_mvn(outputs) -> tfp.distributions.MultivariateNormalTriL:
    """
    :param outputs: (n_samples, 5)
    :return
    """

    # mean of x and y
    x_mean = outputs[:, 0]
    y_mean = outputs[:, 1]

    # std of x and y
    # std is must be 0 or positive
    x_std = tf.exp(outputs[:, 2])
    y_std = tf.exp(outputs[:, 3])

    # correlation coefficient
    # correlation coefficient range is [-1, 1]
    cor = tf.tanh(outputs[:, 4])

    loc = tf.concat([tf.expand_dims(x_mean, axis=1),
                     tf.expand_dims(y_mean, axis=1)], axis=1)

    x_var = tf.square(x_std)
    y_var = tf.square(y_std)
    xy_cor = Multiply()([x_std, y_std, cor])

    cov = tf.stack([x_var, xy_cor, xy_cor, y_var], axis=0)
    cov = tf.transpose(cov, perm=(1, 0))
    cov = tf.reshape(cov, (-1, 2, 2))

    scale = tf.cholesky(cov)
    mvn = tfp.distributions.MultivariateNormalTriL(loc, scale)
    return mvn


def normal2d_log_pdf(outputs, positions):
    """
    :param outputs (n_samples, 5):
    :param positions (n_samples, 2):
    :return: (n_samples,)
    """
    mvn = build_mvn(outputs)
    log_probs = mvn.log_prob(positions)
    return log_probs


def normal2d_sample(outputs):
    """
    :param outputs: (..., 5)
    :return: (tf.reduce_prod(...), 2)
    """
    outputs = tf.reshape(outputs, [-1, 5])
    return build_mvn(outputs).sample()
