import numpy as np
import pandas as pd

from commons.general_utils import DatasetKind, _check_dataset_kind
from preprocessors.preprocess_ewap import EwapPreprocessor
from preprocessors.preprocess_ucy import UcyPreprocessor


def create_dataset_preprocessor(data_dir, dataset_kind):
    dataset_kind = _check_dataset_kind(dataset_kind)
    # ETH dataset
    if dataset_kind in (DatasetKind.eth, DatasetKind.hotel):
        return EwapPreprocessor(data_dir, dataset_kind)

    if dataset_kind in (DatasetKind.zara1, DatasetKind.zara2, DatasetKind.ucy):
        return UcyPreprocessor(data_dir, dataset_kind)

    raise ValueError("dataset_kind")


def thin_out_pos_df(pos_df, interval):
    all_frames = pos_df['frame'].unique()
    remained_frames = np.arange(all_frames[0], all_frames[-1] + 1, interval)
    remained_rows = pos_df['frame'].isin(remained_frames)

    pos_df_thin = pos_df[remained_rows]
    pos_df_thin = pos_df_thin.reset_index(drop=True)
    return pos_df_thin


def interpolate_pos_df(pos_df):
    pos_df_interp = []

    for pid, pid_df in pos_df.groupby('id'):
        observed_frames = np.array(pid_df['frame'])
        frame_range = np.arange(observed_frames[0], observed_frames[-1] + 1)

        x_interp = np.interp(frame_range, pid_df['frame'], pid_df['x'])
        y_interp = np.interp(frame_range, pid_df['frame'], pid_df['y'])

        pos_df_interp.append(pd.DataFrame({
            'frame': frame_range,
            'id': pid,
            'x': x_interp,
            'y': y_interp
        }))

    pos_df_interp = pd.concat(pos_df_interp)
    return pos_df_interp
