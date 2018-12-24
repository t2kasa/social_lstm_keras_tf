from pathlib import Path

import numpy as np
import pandas as pd


def preprocess_ewap(data_dir):
    return EwapPreprocessor(data_dir).preprocess_frame_data()


class EwapPreprocessor:
    """Preprocessor for EWAP dataset."""

    image_sizes = {
        'seq_eth': (640, 480),
        'seq_hotel': (720, 576)
    }

    def __init__(self, data_dir, image_size=None):
        name = Path(data_dir).name
        self.image_size = image_size or self.image_sizes[name]
        if self.image_size is None:
            raise ValueError(f'`image_size` is invalid: {image_size}')

        # read homography matrix
        self.homography = np.genfromtxt(str(Path(data_dir, "H.txt")))
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
    obsmat_file = str(Path(data_dir, "obsmat.txt"))
    obs_columns = ["frame", "id", "px", "pz", "py", "vx", "vz", "vy"]
    obs_df = pd.DataFrame(np.genfromtxt(obsmat_file), columns=obs_columns)
    pos_df = obs_df[["frame", "id", "px", "py"]]
    return pos_df
