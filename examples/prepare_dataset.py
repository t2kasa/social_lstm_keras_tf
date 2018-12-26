import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append((Path(__file__).parent / '..').absolute().as_posix())  # NOQA

from social_lstm_tf.datasets import download_and_arrange_datasets


def _load_args():
    default_root = (Path(__file__).parent / '../data').absolute().as_posix()
    parser = ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default=default_root)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = _load_args()
    download_and_arrange_datasets(args.data_root_dir)


if __name__ == '__main__':
    main()
