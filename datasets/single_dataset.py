from functools import reduce

import numpy as np


def extract_sequences(frame_df, seq_len):
    """Extracts sequences as a dataset.

    :param frame_df: tabled pedestrian positions data. it is expected that the
        data frame has four columns 'frame', 'id', 'x', and 'y'.
    :param seq_len: each sequence length.
    :return: [t, t + seq_len) sequences.
    """
    sequences = []
    all_frames = frame_df['frame'].unique()
    for i in range(len(all_frames) - seq_len + 1):
        frame_range = all_frames[i:i + seq_len]
        df = frame_df[frame_df['frame'].isin(frame_range)]

        # collect pedestrian ids when the pedestrians exist in the all frames
        target_pids = _extract_pids_in_all_frames(df)
        # skip when there are no pedestrians
        if not target_pids:
            continue

        curr_target_df = df[df['id'].isin(target_pids)]
        # built sequence shape is (seq_len, n_pids, 2)
        curr_seq = _build_sequence(curr_target_df)
        sequences.append(curr_seq)

    return sequences


class SingleDataset:
    def __init__(self, frame_data, seq_len):
        self.seq_len = seq_len
        self.x_data, self.y_data = self._build_data(frame_data)

    def _build_data(self, frame_data):
        curr_seqs = []
        next_seqs = []
        all_frames = frame_data['frame'].unique()
        for i in range(len(all_frames) - self.seq_len):
            curr_frames = all_frames[i:i + self.seq_len]
            next_frames = all_frames[i + 1:i + self.seq_len + 1]

            curr_df = frame_data[frame_data['frame'].isin(curr_frames)]
            next_df = frame_data[frame_data['frame'].isin(next_frames)]

            # collect pedestrian ids when the pedestrians exist in the all
            # frames in the current and next sequence
            curr_pids = _extract_pids_in_all_frames(curr_df)
            next_pids = _extract_pids_in_all_frames(next_df)

            target_pids = list(curr_pids & next_pids)
            # skip when there are no pedestrians
            if not target_pids:
                continue

            curr_target_df = curr_df[curr_df['id'].isin(target_pids)]
            next_target_df = next_df[next_df['id'].isin(target_pids)]

            # built sequence shape is (seq_len, n_peds, 2)
            curr_seq = _build_sequence(curr_target_df)
            next_seq = _build_sequence(next_target_df)

            curr_seqs.append(curr_seq)
            next_seqs.append(next_seq)

        return curr_seqs, next_seqs

    def get_data(self, lstm_state_dim):
        zeros_data = np.zeros((len(self.x_data), self.seq_len,
                               self.max_n_peds, lstm_state_dim), np.float32)

        return self.x_data, self.y_data, self.grid_data, zeros_data


def _extract_pids_in_all_frames(df):
    pids = set(
        reduce(np.intersect1d, [g['id'] for _, g in df.groupby('frame')]))
    return pids


def _build_sequence(target_df):
    seq = np.array([np.array(df[['x', 'y']]) for _, df in
                    target_df.groupby('frame')])
    return seq
