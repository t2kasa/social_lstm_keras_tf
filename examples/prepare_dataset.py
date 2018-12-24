import os

from datasets.utils.download_datasets import download_datasets

if __name__ == '__main__':
    here = os.path.abspath(os.path.dirname(__file__))
    data_root_dir = os.path.join(here, '../data')
    download_datasets(data_root_dir)
