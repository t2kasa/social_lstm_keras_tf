from pathlib import Path

from datasets.single_dataset import SingleDataset
from commons.general_utils import DatasetKind
from preprocessors.preprocess_ewap import EwapPreprocessor


def main():
    data_dir = str(Path(__file__).parent / '../data/datasets/eth/hotel')
    dataset_kind = DatasetKind.hotel

    preprocessor = EwapPreprocessor(data_dir, dataset_kind)

    frame_data = preprocessor.preprocess_frame_data()

    dataset = SingleDataset(frame_data, seq_len=3, max_n_peds=52,
                            n_neighbor_pixels=32, grid_side=4,
                            image_size=(720, 576))


if __name__ == '__main__':
    main()
