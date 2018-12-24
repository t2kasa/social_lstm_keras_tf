import tempfile
from pathlib import Path

from datasets.utils import download_and_arrange_datasets
from preprocessors.preprocess_ewap import preprocess_ewap


def test_preprocess_ewap():
    with tempfile.TemporaryDirectory() as temp_dir:
        datasets_dir = download_and_arrange_datasets(temp_dir)
        ewap_dirs = [d for d in Path(datasets_dir, 'ewap_dataset').glob('*')
                     if d.is_dir()]
        for ewap_dir in ewap_dirs:
            preprocess_ewap(ewap_dir)
