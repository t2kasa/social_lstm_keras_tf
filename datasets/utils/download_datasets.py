import os
import shutil
import tempfile
import urllib.request

import patoolib

_ewap_url = 'http://www.vision.ee.ethz.ch/datasets_extra/ewap_dataset_light.tgz'
_ucy_univ_url = 'http://graphics.cs.ucy.ac.cy/files/crowds/data/data_zara.rar'
_ucy_zara_url = 'http://graphics.cs.ucy.ac.cy/files/crowds/data/data_university_students.rar'  # NOQA


def download_datasets(data_root_dir):
    datasets_dir = os.path.join(data_root_dir, 'datasets')
    if os.path.isdir(datasets_dir):
        return

    os.makedirs(datasets_dir, exist_ok=True)
    _maybe_download_ewap_dataset(datasets_dir)
    _maybe_download_ucy_dataset(datasets_dir)


def _maybe_download_ewap_dataset(datasets_dir):
    _download_and_unpack(_ewap_url, datasets_dir)


def _maybe_download_ucy_dataset(datasets_dir):
    ucy_dir = os.path.join(datasets_dir, 'ucy_dataset')
    _download_and_unpack(_ucy_zara_url, ucy_dir)
    _download_and_unpack(_ucy_univ_url, ucy_dir)


def _download_and_unpack(content_url, out_dir):
    _, ext = os.path.splitext(content_url)

    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_file = os.path.join(temp_dir, os.path.basename(content_url))
        urllib.request.urlretrieve(content_url, downloaded_file)

        if ext.lower() == '.rar':
            patoolib.extract_archive(downloaded_file, -1, out_dir)
        else:
            shutil.unpack_archive(downloaded_file, out_dir)
