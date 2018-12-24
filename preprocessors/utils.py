import numpy as np
import pandas as pd


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
