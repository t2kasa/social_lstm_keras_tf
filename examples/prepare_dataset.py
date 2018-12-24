import os

from datasets.utils import download_and_arrange_datasets

if __name__ == '__main__':
    here = os.path.abspath(os.path.dirname(__file__))
    data_root_dir = os.path.join(here, '../data')
    download_and_arrange_datasets(data_root_dir)
