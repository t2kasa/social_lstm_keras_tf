import shutil
import tempfile
import urllib.request
from pathlib import Path

import patoolib

_ewap_url = 'http://www.vision.ee.ethz.ch/datasets_extra/ewap_dataset_light.tgz'
_ucy_univ_url = 'http://graphics.cs.ucy.ac.cy/files/crowds/data/data_zara.rar'
_ucy_zara_url = 'http://graphics.cs.ucy.ac.cy/files/crowds/data/data_university_students.rar'  # NOQA
_ucy_file_names = ['crowds_zara01.vsp', 'crowds_zara02.vsp', 'students003.vsp']


def download_and_arrange_datasets(data_root_dir):
    datasets_dir = Path(data_root_dir, 'datasets')
    if datasets_dir.exists():
        return
    datasets_dir.mkdir(parents=True, exist_ok=True)

    _download_and_arrange_ewap_dataset(datasets_dir)
    _download_and_arrange_ucy_dataset(datasets_dir)

    return datasets_dir


def _download_and_arrange_ewap_dataset(datasets_dir):
    _download_and_unpack(_ewap_url, datasets_dir)


def _download_and_arrange_ucy_dataset(datasets_dir):
    ucy_dir = Path(datasets_dir, 'ucy_dataset')
    _download_and_unpack(_ucy_zara_url, ucy_dir)
    _download_and_unpack(_ucy_univ_url, ucy_dir)

    # move actually used files to corresponding named directories
    for name in _ucy_file_names:
        dst_dir = Path(ucy_dir, Path(name).stem)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(Path(ucy_dir, name), Path(dst_dir, name))

    # remove unused files
    for f in ucy_dir.glob('*.*'):
        f.unlink()


def _download_and_unpack(content_url, out_dir):
    ext = Path(content_url).suffix

    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_file = str(Path(temp_dir, Path(content_url).name))
        urllib.request.urlretrieve(content_url, downloaded_file)

        if ext.lower() == '.rar':
            patoolib.extract_archive(downloaded_file, -1, out_dir)
        else:
            shutil.unpack_archive(downloaded_file, out_dir)
