import tempfile
from pathlib import Path

from datasets.utils import download_and_arrange_datasets
from preprocessors.preprocess_ucy import preprocess_ucy


def test_ucy_preprocessor_init():
    with tempfile.TemporaryDirectory() as temp_dir:
        datasets_dir = download_and_arrange_datasets(temp_dir)
        ucy_dirs = Path(datasets_dir, 'ucy_dataset').glob('*')
        for ucy_dir in ucy_dirs:
            preprocess_ucy(str(ucy_dir))
