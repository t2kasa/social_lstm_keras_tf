import numpy as np
import tensorflow as tf


def compute_social_tensor(positions, hidden_states, cell_side: float,
                          n_side_cells: int):
    """Computes social tensor.

    :param positions: (n_pedestrians, 2).
        The pedestrian positions.
    :param hidden_states: (n_pedestrians, n_states).
        The hidden states of LSTM.
    :param cell_side: side of one cell.
    :param n_side_cells: The number of cells tiled on the grid side. That is,
        the tiles on the grid is `n_grid_cells ** 2` tiles.
    :return: (n_pedestrians, n_grid_cells, n_grid_cells, n_states)
        social tensors.
    """
    n_pedestrians = tf.shape(positions).numpy()[0]
    n_states = tf.shape(hidden_states).numpy()[1]
    n_half_side_cells = n_side_cells // 2

    cell_borders = tf.linspace(-cell_side * n_half_side_cells,
                               cell_side * n_half_side_cells,
                               n_side_cells + 1)

    indices = np.arange(tf.shape(positions).numpy()[0])

    social_tensors = []
    # Compute social tensors.
    # i-th loop corresponds to the i-th pedestrian.
    for i in range(n_pedestrians):
        position_i = tf.boolean_mask(positions, indices == i)
        other_pos = tf.boolean_mask(positions, indices != i)
        other_hidden_states = tf.boolean_mask(hidden_states, indices != i)

        # First remain only neighbor pedestrians.

        pos_diff = other_pos - position_i
        # so bucketize() in tensorflow raises an error,
        # use digitize() in numpy as workaround.
        cell_xy_indices = np.digitize(pos_diff.numpy(),
                                      cell_borders.numpy()) - 1
        neighbor_mask = tf.reduce_all(
            tf.logical_and(0 <= cell_xy_indices,
                           cell_xy_indices < n_side_cells),
            axis=1)

        neighbor_xy_indices = tf.boolean_mask(cell_xy_indices, neighbor_mask)
        neighbor_hidden_states = tf.boolean_mask(other_hidden_states,
                                                 neighbor_mask)

        social_tensor_i = [[] for _ in range(n_side_cells ** 2)]
        for xy_index, h in zip(neighbor_xy_indices, neighbor_hidden_states):
            cell_index = xy_index[1] * n_side_cells + xy_index[0]
            social_tensor_i[cell_index].append(h)

        social_tensor_i = [tf.reduce_sum(s, axis=0) if len(s) != 0
                           else tf.zeros(n_states) for s in social_tensor_i]
        social_tensor_i = tf.reshape(tf.stack(social_tensor_i, axis=0),
                                     (n_side_cells, n_side_cells, n_states))

        social_tensors.append(social_tensor_i)

    social_tensors = tf.stack(social_tensors, axis=0)
    return social_tensors
