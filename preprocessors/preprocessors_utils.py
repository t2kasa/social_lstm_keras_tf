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
