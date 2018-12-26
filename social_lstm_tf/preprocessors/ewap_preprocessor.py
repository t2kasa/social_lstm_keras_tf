from pathlib import Path

import numpy as np
import pandas as pd

from social_lstm_tf.preprocessors.utils import interpolate_pos_df
from social_lstm_tf.preprocessors.utils import thin_out_pos_df


class EwapPreprocessor:
    """Preprocessor for EWAP dataset."""

    image_sizes = {
        'seq_eth': (640, 480),
        'seq_hotel': (720, 576)
    }

    frame_interval = 10

    def __init__(self, data_dir, image_size=None):
        name = Path(data_dir).name
        self.image_size = image_size or self.image_sizes[name]
        if self.image_size is None:
            raise ValueError(f'`image_size` is invalid: {image_size}')

        # read homography matrix
        self.homography = _read_homography_file(data_dir)
        # read trajectory data
        self.raw_pos_df = _read_obsmat_file(data_dir)

    def preprocess_frame_data(self):
        pos_df_pre = interpolate_pos_df(self.raw_pos_df)
        pos_df_pre = thin_out_pos_df(pos_df_pre, self.frame_interval)
        pos_df_pre = self._normalize_pos_df(pos_df_pre)
        pos_df_pre = pos_df_pre.sort_values(['frame', 'id'])
        return pos_df_pre

    def _normalize_pos_df(self, pos_df):
        xy = np.array(pos_df[["x", "y"]])
        xy = self._world_to_image_xy(xy, self.homography)
        xy = xy / self.image_size

        pos_df_norm = pd.DataFrame({
            "frame": pos_df["frame"],
            "id": pos_df["id"],
            "x": xy[:, 0],
            "y": xy[:, 1]
        })
        return pos_df_norm

    @staticmethod
    def _world_to_image_xy(world_xy, homography):
        """Converts world (x, y) position to image (x, y) position.

        This function use inverse mapping of homography transform.

        :param world_xy: world (x, y) positions
        :param homography: homography matrix
        :return: image (x, y) positions
        """
        world_xy = np.array(world_xy)
        world_xy1 = np.concatenate([world_xy, np.ones((len(world_xy), 1))],
                                   axis=1)
        image_xy1 = np.linalg.inv(homography).dot(world_xy1.T).T
        image_xy = image_xy1[:, :2] / np.expand_dims(image_xy1[:, 2], axis=1)
        return image_xy


def _read_homography_file(data_dir):
    return np.genfromtxt(str(Path(data_dir, "H.txt")))


def _read_obsmat_file(data_dir):
    obs_cols = ["frame", "id", "px", "pz", "py", "vx", "vz", "vy"]
    obsmat_file = str(Path(data_dir, "obsmat.txt"))
    obs_df = pd.DataFrame(np.genfromtxt(obsmat_file), columns=obs_cols)

    pos_df = obs_df[["frame", "id", "px", "py"]]
    pos_df.columns = ["frame", "id", "x", "y"]
    return pos_df
