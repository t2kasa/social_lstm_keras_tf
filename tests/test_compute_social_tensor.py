import tensorflow as tf

from social_lstm.grid_tfe import compute_social_tensor


def test_compute_social_tensor():
    tf.enable_eager_execution()
    # prepare dummy data
    positions = tf.constant([
        [0.5, 0.5],
        [0.125, 0.125],
        [0.25, 0.25],
        [0.25, 0.75],
        [10.0, 10.0]], dtype=tf.float32)
    hidden_states = tf.constant([
        [0.5, 0.5],
        [0.125, 0.125],
        [0.25, 0.25],
        [0.25, 0.75],
        [0.75, 0.25]], dtype=tf.float32)

    cell_side = 0.5
    n_grid_cells = 2

    social_tensors = compute_social_tensor(positions, hidden_states, cell_side,
                                           n_grid_cells)
    expected_shape = tf.constant([5, 2, 2, 2])
    expected_first_social_tensor = tf.constant([[[0.375, 0.375],
                                                 [0.0, 0.0]],
                                                [[0.25, 0.75],
                                                 [0.0, 0.0]]])
    tf.assert_equal(tf.shape(social_tensors), expected_shape)
    tf.assert_equal(social_tensors[0], expected_first_social_tensor)
