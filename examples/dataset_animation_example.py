import matplotlib

# to show animation in PyCharm
matplotlib.use('Qt5Agg')  # NOQA

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib.animation import FuncAnimation
from datasets.load_single_dataset import load_single_dataset


class TrajectoryAnimator:
    def __init__(self, ds):
        self.ds = ds
        self.iter = ds.make_one_shot_iterator()
        self.fig, self.axes = plt.subplots(1, 1)

    def update(self, t):
        self.axes.clear()

        obs, pred = self.iter.get_next()
        obs, pred = obs[0].numpy(), pred[0].numpy()
        obs = np.transpose(obs, axes=(1, 0, 2))
        pred = np.transpose(pred, axes=(1, 0, 2))

        for o in obs:
            self.axes.plot(o[:, 0], o[:, 1])
        for p in pred:
            self.axes.plot(p[:, 0], p[:, 1])

        return self.axes,


def main():
    tf.enable_eager_execution()

    parser = ArgumentParser()
    parser.add_argument('--data_dirs', type=str, nargs='+')
    data_dirs = parser.parse_args().data_dirs

    ds = load_single_dataset(data_dirs, 8, 12)
    animator = TrajectoryAnimator(ds)
    ani = FuncAnimation(animator.fig, animator.update, interval=500)
    plt.show()


if __name__ == '__main__':
    main()
