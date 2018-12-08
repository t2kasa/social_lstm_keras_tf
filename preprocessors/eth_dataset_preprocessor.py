import os

import numpy as np
import pandas as pd

from general_utils import get_image_size


class EthDatasetPreprosessor:
    """Preprocessor for ETH dataset."""

    def __init__(self, data_dir, dataset_kind):
        self._data_dir = data_dir
        self.image_size = get_image_size(dataset_kind)

        # read homography matrix
        homography_file = os.path.join(self.data_dir, "H.txt")
        self.homography = np.genfromtxt(homography_file)

        # read trajectory data
        obsmat_file = os.path.join(self.data_dir, "obsmat.txt")
        obs_columns = ["frame", "id", "px", "pz", "py", "vx", "vz", "vy"]
        obs_df = pd.DataFrame(np.genfromtxt(obsmat_file), columns=obs_columns)
        # remain only (frame index, pedestrian id, position x, position y)
        self.raw_df = obs_df[["frame", "id", "px", "py"]]

    def preprocess_frame_data(self):
        # position preprocessing
        xy = np.array(self.raw_df[["px", "py"]])

        # world xy to image xy: inverse mapping of homography
        xy = self._world_to_image_xy(xy, self.homography)

        # normalize
        xy = xy / self.image_size

        # construct preprocessed df
        pos_df_preprocessed = pd.DataFrame({
            "frame": self.raw_df["frame"],
            "id": self.raw_df["id"],
            "x": xy[:, 0],
            "y": xy[:, 1]
        })

        return pos_df_preprocessed

    @property
    def data_dir(self):
        return self._data_dir

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
