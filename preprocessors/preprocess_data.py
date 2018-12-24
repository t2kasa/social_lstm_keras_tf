from pathlib import Path

from preprocessors.ewap_preprocessor import EwapPreprocessor
from preprocessors.ucy_preprocessor import UcyPreprocessor
from preprocessors.utils import ucy_data_dir_names, ewap_data_dir_names


def preprocess_data(data_dir):
    name = Path(data_dir).name
    if name in ucy_data_dir_names:
        return preprocess_ucy(data_dir)
    elif name in ewap_data_dir_names:
        return preprocess_ewap(data_dir)
    else:
        raise ValueError(f'`data_dir` is invalid: {data_dir}')


def preprocess_ewap(data_dir):
    return EwapPreprocessor(data_dir).preprocess_frame_data()


def preprocess_ucy(data_dir):
    return UcyPreprocessor(data_dir).preprocess_frame_data()
