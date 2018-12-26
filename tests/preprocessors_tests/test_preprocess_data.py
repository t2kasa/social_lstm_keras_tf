import tempfile
from pathlib import Path

from datasets import download_and_arrange_datasets
from preprocessors import preprocess_data
from preprocessors.preprocess_data import preprocess_data


def test_preprocess_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        datasets_dir = download_and_arrange_datasets(temp_dir)
        ewap_dirs = [d for d in Path(datasets_dir, 'ewap_dataset').glob('*')
                     if d.is_dir()]
        for ewap_dir in ewap_dirs:
            preprocess_data(ewap_dir)

        ucy_dirs = [d for d in Path(datasets_dir, 'ucy_dataset').glob('*')
                    if d.is_dir()]
        for ucy_dir in ucy_dirs:
            preprocess_data(ucy_dir)
