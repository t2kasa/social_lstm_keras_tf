from pathlib import Path

import numpy as np
import pandas as pd

from preprocessors.utils import interpolate_pos_df
from preprocessors.utils import thin_out_pos_df


def preprocess_ucy(data_dir):
    return UcyPreprocessor(data_dir).preprocess_frame_data()


class UcyPreprocessor:
    ped_start_line_words = 'Num of control points'
    # _1, _2, _3, and _4 are dummy columns, which are not used.
    vsp_cols = ['x', 'y', 'frame', 'gaze', '_1', '_2', '_3', '_4']

    image_sizes = {
        'crowds_zara01': (720, 576),
        'crowds_zara02': (720, 576),
        'students003': (720, 576)
    }

    # arrange frame interval
    frame_interval = 10

    def __init__(self, data_dir, image_size=None):
        name = Path(data_dir).name
        self._data_dir = data_dir
        self.image_size = image_size or self.image_sizes.get(name)
        if self.image_size is None:
            raise ValueError(f'`image_size` is invalid: {image_size}')

    def preprocess_frame_data(self):
        lines = self._read_lines(self._get_vsp_file(self._data_dir))

        # find first lines specifying each pedestrian trajectory
        ped_start_indices = [li for li, line in enumerate(lines) if
                             line.find(self.ped_start_line_words) != -1]

        pos_df_raw = []

        # extract pedestrian positions as a data frame
        for i, start_index in enumerate(ped_start_indices):
            n_pos_i = int(lines[start_index].split()[0])
            pos_lines_i = lines[start_index + 1:start_index + 1 + n_pos_i]
            pos_df_raw_i = pd.DataFrame([line.split() for line in pos_lines_i],
                                        columns=self.vsp_cols)
            # in UCY dataset, pedestiran 'id' is not given,
            # therefore add 'id' column with serial number.
            pos_df_raw_i['id'] = i + 1

            pos_df_raw.append(pos_df_raw_i)

        pos_df_raw = pd.concat(pos_df_raw)
        pos_df_raw = pos_df_raw[['frame', 'id', 'x', 'y']].astype(np.float32)
        pos_df_raw = pos_df_raw.reset_index(drop=True)

        # interpolate, thin out, and normalize
        pos_df_pre = interpolate_pos_df(pos_df_raw)
        pos_df_pre = thin_out_pos_df(pos_df_pre, self.frame_interval)
        pos_df_pre = self.normalize_pos_df(pos_df_pre, self.image_size)
        pos_df_pre = pos_df_pre.sort_values(['frame', 'id'])

        return pos_df_pre

    @staticmethod
    def _get_vsp_file(data_dir):
        return str(Path(data_dir, f'{Path(data_dir).name}.vsp'))

    @staticmethod
    def _read_lines(file):
        with open(file, 'r') as f:
            return f.readlines()

    @staticmethod
    def normalize_pos_df(pos_df, image_size):
        image_size = np.array(image_size)

        xy = np.array(pos_df[['x', 'y']])
        # originally (0, 0) is the center of the frame,
        # therefore move (0, 0) to top-left
        xy += image_size / 2
        # clipping
        xy[:, 0] = np.clip(xy[:, 0], 0.0, image_size[0] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0.0, image_size[1] - 1)

        # normalize
        xy /= image_size

        # normalize position (x, y) respectively
        pos_df_norm = pd.DataFrame({
            'frame': pos_df['frame'],
            'id': pos_df['id'],
            'x': xy[:, 0],
            'y': xy[:, 1]
        })
        return pos_df_norm
