import os
import shutil
import tempfile
import urllib.request

_ewap_url = 'http://www.vision.ee.ethz.ch/datasets_extra/ewap_dataset_light.tgz'


def download_datasets(data_root_dir):
    datasets_dir = os.path.join(data_root_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    _maybe_download_ewap_dataset(datasets_dir)


def _maybe_download_ewap_dataset(datasets_dir):
    # return when you have already downloaded ewap dataset.
    ewap_dir = os.path.join(datasets_dir, 'ewap_dataset')
    if os.path.isdir(ewap_dir):
        return

    # download ewap dataset, then unpack to the datasets directory.
    _download_and_unpack(_ewap_url, datasets_dir)


def _download_and_unpack(content_url, out_dir):
    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_file = os.path.join(temp_dir, os.path.basename(content_url))
        urllib.request.urlretrieve(content_url, downloaded_file)
        shutil.unpack_archive(downloaded_file, out_dir)
