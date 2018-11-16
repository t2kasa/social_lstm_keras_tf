import numpy as np
import tensorflow as tf

from tfe_normal_sampler import build_mvn


def test_build_mvn():
    tf.enable_eager_execution()
    n_samples = 3
    out_dim = 5
    outputs = tf.random.normal((n_samples, out_dim))
    mvn = build_mvn(outputs)

    expected_batch_shape = np.array([3])
    actual_batch_shape = mvn.batch_shape_tensor().numpy()
    np.testing.assert_equal(expected_batch_shape, actual_batch_shape)
