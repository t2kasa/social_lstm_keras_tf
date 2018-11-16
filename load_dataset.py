from functools import reduce

import numpy as np

from general_utils import get_data_dir
from general_utils import get_image_size
from grid import grid_mask
from load_model_config import ModelConfig
from preprocessors.preprocessors_utils import create_dataset_preprocessor


def load_dataset_from_config(config: ModelConfig):
    data_dir = get_data_dir(config.data_root, config.test_dataset_kind)

    return load_dataset(data_dir=data_dir,
                        dataset_kind=config.test_dataset_kind,
                        seq_len=config.obs_len + config.pred_len,
                        max_n_peds=config.max_n_peds,
                        n_neighbor_pixels=config.n_neighbor_pixels,
                        grid_side=config.grid_side)


def load_dataset(data_dir, dataset_kind, seq_len, max_n_peds, n_neighbor_pixels,
                 grid_side):
    loader = _SingleDatasetLoader(data_dir, seq_len, max_n_peds,
                                  n_neighbor_pixels, grid_side,
                                  dataset_kind)
    dataset = loader.load()
    return dataset


class SingleDataset:
    def __init__(self, frame_data, seq_len, max_n_peds, n_neighbor_pixels,
                 grid_side, image_size):
        self.seq_len = seq_len
        self.max_n_peds = max_n_peds
        self.n_neighbor_pixels = n_neighbor_pixels
        self.grid_side = grid_side
        self.image_size = image_size

        self.x_data, self.y_data, self.grid_data = self._build_data(frame_data)

    def _build_data(self, frame_data):
        x_data = []
        y_data = []

        for i in range(len(frame_data) - self.seq_len):
            cf_data = frame_data[i:i + self.seq_len, ...]
            nf_data = frame_data[i + 1:i + self.seq_len + 1, ...]

            pid_col = 0
            # collect pedestrian ids where the pedestrians exist in the
            # all frames of the current and next sequence
            cf_ped_ids = reduce(
                set.intersection,
                [set(cf_pids) for cf_pids in cf_data[..., pid_col]])
            nf_ped_ids = reduce(
                set.intersection,
                [set(nf_pids) for nf_pids in nf_data[..., pid_col]])

            ped_ids = list(cf_ped_ids & nf_ped_ids - {0})
            # if there are no pedestrians
            if not ped_ids:
                continue

            x = np.zeros((self.seq_len, self.max_n_peds, 3))
            y = np.zeros((self.seq_len, self.max_n_peds, 3))

            # fi = frame index, cf = current frame, nf = next frame
            for fi, (cf, nf) in enumerate(zip(cf_data, nf_data)):
                for j, ped_id in enumerate(ped_ids):
                    cf_ped_row = cf[:, 0] == ped_id
                    nf_ped_row = nf[:, 0] == ped_id

                    if np.any(cf_ped_row):
                        x[fi, j, :] = cf[cf[:, 0] == ped_id]
                    if np.any(nf_ped_row):
                        y[fi, j, :] = nf[nf[:, 0] == ped_id]

            x_data.append(x)
            y_data.append(y)

        # compute grid mask
        grid_data = [grid_mask(x, self.image_size, self.n_neighbor_pixels,
                               self.grid_side) for x in x_data]

        data_tuple = (np.array(x_data, np.float32),
                      np.array(y_data, np.float32),
                      np.array(grid_data, np.float32))
        return data_tuple

    def get_data(self, lstm_state_dim):
        zeros_data = np.zeros((len(self.x_data), self.seq_len,
                               self.max_n_peds, lstm_state_dim), np.float32)

        return self.x_data, self.y_data, self.grid_data, zeros_data


class _SingleDatasetLoader:
    def __init__(self, data_dir, seq_len, max_n_peds, n_neighbor_pixels,
                 grid_side, dataset_kind):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.max_n_peds = max_n_peds
        self.n_neighbor_pixels = n_neighbor_pixels
        self.grid_side = grid_side
        self.dataset_kind = dataset_kind
        self.image_size = get_image_size(dataset_kind)

    def load(self) -> SingleDataset:
        preprocessor = create_dataset_preprocessor(self.data_dir,
                                                   self.dataset_kind)
        df = preprocessor.preprocess_frame_data()

        # All frame IDs in the current dataset
        all_frames = df["frame"].unique().tolist()
        n_all_frames = len(all_frames)

        all_frame_data = np.zeros((n_all_frames, self.max_n_peds, 3),
                                  np.float64)
        for index, frame in enumerate(all_frames):
            peds_with_pos = np.array(df[df["frame"] == frame][["id", "x", "y"]])

            n_peds = len(peds_with_pos)

            all_frame_data[index, 0:n_peds, :] = peds_with_pos

        return SingleDataset(all_frame_data, self.seq_len,
                             self.max_n_peds, self.n_neighbor_pixels,
                             self.grid_side, self.image_size)
