import os

import numpy as np
import pandas as pd

from commons.general_utils import get_image_size


class EthPreprosessor:
    """Preprocessor for ETH dataset."""

    def __init__(self, data_dir, dataset_kind):
        self.image_size = get_image_size(dataset_kind)
        # read homography matrix
        self.homography = np.genfromtxt(os.path.join(data_dir, "H.txt"))
        # read trajectory data
        self.raw_pos_df = _load_obsmat_file(data_dir)

    def preprocess_frame_data(self):
        # position preprocessing
        xy = np.array(self.raw_pos_df[["px", "py"]])
        # world xy to image xy: inverse mapping of homography
        xy = self._world_to_image_xy(xy, self.homography)
        # normalize
        xy = xy / self.image_size

        preprocessed_df = pd.DataFrame({
            "frame": self.raw_pos_df["frame"],
            "id": self.raw_pos_df["id"],
            "x": xy[:, 0],
            "y": xy[:, 1]
        })
        return preprocessed_df

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


def _load_obsmat_file(data_dir):
    obsmat_file = os.path.join(data_dir, "obsmat.txt")
    obs_columns = ["frame", "id", "px", "pz", "py", "vx", "vz", "vy"]
    obs_df = pd.DataFrame(np.genfromtxt(obsmat_file), columns=obs_columns)
    pos_df = obs_df[["frame", "id", "px", "py"]]
    return pos_df
