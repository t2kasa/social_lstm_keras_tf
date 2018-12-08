from functools import partial

import numpy as np
import tensorflow as tf
from keras.layers import Lambda
from keras.utils import to_categorical


def tf_grid_mask(x, image_size, n_neighbor_pixels, grid_side):
    func = partial(_tf_grid_mask_frame, image_size=image_size,
                   n_neighbor_pixels=n_neighbor_pixels, grid_side=grid_side)

    return Lambda(
        lambda x: tf.stack(tf.map_fn(lambda frame: func(frame), x)))(x)


def grid_mask(x, image_size, n_neighbor_pixels, grid_side):
    func = partial(_grid_mask_frame, image_size=image_size,
                   n_neighbor_pixels=n_neighbor_pixels, grid_side=grid_side)

    return np.array([func(frame) for frame in x])


def _tf_grid_mask_frame(frame, image_size, n_neighbor_pixels, grid_side):
    """Compute a grid mask for TensorFlow.

    :param frame:
    :param image_size:
    :param n_neighbor_pixels:
    :param grid_side:
    :return:
    """
    max_n_peds = frame.shape.as_list()[0]
    pids = frame[:, 0]

    # --------------------
    # compute id_mask
    # --------------------

    def compute_id_mask(pids):
        id_mask = tf.tensordot(tf.expand_dims(pids, axis=1),
                               tf.transpose(tf.expand_dims(pids, axis=1)),
                               axes=(1, 0))
        id_mask = tf.cast(id_mask, tf.bool)
        # mask self-to-self (diagonal elements)
        id_mask = tf.logical_and(
            tf.logical_not(tf.cast(tf.eye(max_n_peds), tf.bool)), id_mask)
        id_mask = tf.expand_dims(id_mask, axis=2)
        id_mask = tf.cast(id_mask, tf.float32)

        return id_mask

    id_mask = Lambda(compute_id_mask)(pids)

    bound = n_neighbor_pixels / np.array(image_size)
    pos = frame[:, 1:]
    tl = pos - bound / 2
    br = pos + bound / 2

    frame_mask = []
    for self_index in range(max_n_peds):
        is_neighbor = Lambda(lambda pos: tf.cast(tf.reduce_all(
            tf.concat([tl[self_index] <= pos, pos < br[self_index]], axis=1),
            axis=1), np.int32))(pos)

        cell_xy = Lambda(lambda pos: tf.cast(
            tf.floor(((pos - tl[self_index]) / bound) * grid_side), tf.int32))(
            pos)

        cell_index = cell_xy[:, 0] + cell_xy[:, 1] * grid_side
        cell_index = cell_index * is_neighbor

        self_frame_mask = tf.stack(tf.map_fn(
            lambda c: tf.eye(grid_side ** 2, dtype=np.int32)[c], cell_index),
            axis=0)
        self_frame_mask *= tf.expand_dims(is_neighbor, 1)
        frame_mask.append(self_frame_mask)

    frame_mask = tf.stack(frame_mask, axis=0)
    frame_mask = tf.cast(frame_mask, tf.float32)
    # mask not exist elements & self-to-self pair
    frame_mask *= id_mask
    return frame_mask


def _grid_mask_frame(frame, image_size, n_neighbor_pixels, grid_side):
    """
    This function computes the binary mask that represents the
    occupancy of each ped in the other"s grid
    params:
    frame : (max_n_peds, 3)
    image_size : [width, height]
    n_neighbor_pixels : considered neighborhood pixels
    grid_side : Scalar value representing the size of the grid discretization
    """
    # Maximum number of pedestrians
    max_n_peds = frame.shape[0]

    # compute mask array based on pid
    pids = frame[:, 0]
    id_mask = np.dot(np.expand_dims(pids, axis=1),
                     np.expand_dims(pids, axis=1).T)
    id_mask[id_mask > 0] = 1
    # mask self-to-self (diagonal elements)
    id_mask *= ~np.eye(max_n_peds).astype(np.bool)
    id_mask = np.expand_dims(id_mask, axis=2)

    bound = n_neighbor_pixels / np.array(image_size)
    pos = frame[:, 1:]
    tl = pos - bound / 2
    br = pos + bound / 2

    frame_mask = []
    for self_index in range(max_n_peds):
        is_neighbor = np.all(
            np.concatenate([tl[self_index] <= pos, pos < br[self_index]],
                           axis=1),
            axis=1)

        cell_xy = np.floor(((pos - tl[self_index]) / bound) * grid_side).astype(
            np.int)
        cell_index = cell_xy[:, 0] + cell_xy[:, 1] * grid_side
        cell_index *= is_neighbor

        self_frame_mask = to_categorical(cell_index, grid_side ** 2)
        self_frame_mask *= np.expand_dims(is_neighbor, 1)
        frame_mask.append(self_frame_mask)

    frame_mask = np.stack(frame_mask, axis=0)
    # mask not exist elements & self-to-self pair
    frame_mask *= id_mask

    frame_mask_legacy = _legacy_grid_mask_frame(frame, image_size,
                                                n_neighbor_pixels,
                                                grid_side)
    assert np.array_equal(frame_mask, frame_mask_legacy)
    return frame_mask


def _legacy_grid_mask_frame(frame, image_size, n_neighbor_pixels, grid_side):
    max_n_peds = frame.shape[0]
    width, height = image_size[0], image_size[1]

    frame_mask = np.zeros((max_n_peds, max_n_peds, grid_side ** 2))

    width_bound = n_neighbor_pixels / width
    height_bound = n_neighbor_pixels / height

    # For each ped in the frame (existent and non-existent)
    for self_index in range(max_n_peds):
        # If pedID is zero, then non-existent ped
        if frame[self_index, 0] == 0:
            # Binary mask should be zero for non-existent ped
            continue

        # Get x and y of the current ped
        self_x, self_y = frame[self_index, 1], frame[self_index, 2]

        tl_x, br_x = self_x - width_bound / 2, self_x + width_bound / 2
        tl_y, br_y = self_y - height_bound / 2, self_y + height_bound / 2

        # For all the other peds
        for other_index in range(max_n_peds):
            # If other pedID is zero, then non-existent ped
            if frame[other_index, 0] == 0:
                # Binary mask should be zero
                continue

            # If the other pedID is the same as current pedID
            if frame[other_index, 0] == frame[self_index, 0]:
                # The ped cannot be counted in his own grid
                continue

            # Get x and y of the other ped
            other_x, other_y = frame[other_index, 1], frame[other_index, 2]
            if other_x >= br_x or other_x < tl_x \
                    or other_y >= br_y or other_y < tl_y:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            cell_x = int(np.floor(((other_x - tl_x) / width_bound) * grid_side))
            cell_y = int(
                np.floor(((other_y - tl_y) / height_bound) * grid_side))

            # Other ped is in the corresponding grid cell of current ped
            frame_mask[self_index, other_index, cell_x + cell_y * grid_side] = 1

    return frame_mask
