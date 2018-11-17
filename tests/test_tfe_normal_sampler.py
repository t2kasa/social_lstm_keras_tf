import tensorflow as tf

from tfe_normal_sampler import build_mvn


def test_build_mvn():
    tf.enable_eager_execution()
    n_samples = 3
    out_dim = 5
    outputs = tf.random.normal((n_samples, out_dim))
    mvn = build_mvn(outputs)

    actual_batch_shape = mvn.batch_shape_tensor().numpy()
    tf.assert_equal(actual_batch_shape, tf.constant([3]))
