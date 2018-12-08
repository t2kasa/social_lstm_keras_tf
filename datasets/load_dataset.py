import numpy as np

from datasets.single_dataset import SingleDataset
from commons.general_utils import get_data_dir
from commons.general_utils import get_image_size
from commons.load_model_config import ModelConfig
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
