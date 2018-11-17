import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

states_dim = 2

hidden_states = np.array([
    [0.5, 0.5],
    [0.125, 0.125],
    [0.25, 0.25],
    [0.25, 0.75],
    [0.75, 0.25]])

pos = np.array([
    [0.5, 0.5],
    [0.125, 0.125],
    [0.25, 0.25],
    [0.25, 0.75],
    [10.0, 10.0]])

hidden_states = tf.convert_to_tensor(hidden_states, dtype=tf.float32)
pos = tf.convert_to_tensor(pos, dtype=tf.float32)

cell_side = 0.5
# Construct (n_grid_cells, n_grid_cells) grid.
n_grid_cells = 2
n_half_grid_cells = n_grid_cells // 2

cell_borders = tf.linspace(-cell_side * n_half_grid_cells,
                           cell_side * n_half_grid_cells,
                           n_grid_cells + 1)

indices = np.arange(tf.shape(pos).numpy()[0])

i = 0
pos_i = tf.boolean_mask(pos, indices == i)
other_pos = tf.boolean_mask(pos, indices != i)
other_hidden_states = tf.boolean_mask(hidden_states, indices != i)

# Social tensor computation
# remain only neighbor pedestrians

pos_diff = other_pos - pos_i
# so bucketize() in tensorflow raises an error,
# use digitize() in numpy as workaround.
cell_xy_indices = np.digitize(pos_diff.numpy(), cell_borders.numpy()) - 1
neighbor_mask = tf.reduce_all(
    tf.logical_and(0 <= cell_xy_indices, cell_xy_indices < n_grid_cells),
    axis=1)

neighbor_pos_diff = tf.boolean_mask(pos_diff, neighbor_mask)
neighbor_xy_indices = tf.boolean_mask(cell_xy_indices, neighbor_mask)
neighbor_hidden_states = tf.boolean_mask(other_hidden_states, neighbor_mask)

# cell_xy_indices = math_ops.bucketize(pos_diff, cell_borders)


social_tensor = [[] for _ in range(n_grid_cells ** 2)]
for xy_index, h in zip(neighbor_xy_indices, neighbor_hidden_states):
    cell_index = xy_index[1] * n_grid_cells + xy_index[0]
    social_tensor[cell_index].append(h)

social_tensor = [tf.reduce_sum(s, axis=0) if len(s) != 0
                 else tf.zeros(states_dim) for s in social_tensor]
social_tensor = tf.reshape(tf.stack(social_tensor, axis=0),
                           (-1, n_grid_cells, n_grid_cells))


def compute_social_tensor(cell_side, n_grid_cells):
    """Computes social tensor of pedestrians.

    :param cell_side: The cell side of the grid.
    :param n_grid_cells: The number of cells of the grid side.
    :return:
    """
    pass
