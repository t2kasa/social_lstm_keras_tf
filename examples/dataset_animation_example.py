import matplotlib

# to show animation in PyCharm
matplotlib.use('Qt5Agg')  # NOQA

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from commons.general_utils import DatasetKind
from preprocessors.eth_preprocessor import EthPreprosessor


class TrajectoryAnimator:
    def __init__(self, df):
        self.df = df
        self.fig, self.axes = plt.subplots(1, 1)
        self.frames = self.df['frame'].unique()

    def update(self, t):
        self.axes.clear()

        frame_range = self.frames[np.maximum(t - 10, 0):t]
        target_df = self.df[self.df['frame'].isin(frame_range)]
        print(np.unique(target_df['frame']))

        for i, df_i in target_df.groupby('id'):
            xy = np.array(df_i[['x', 'y']])
            print(xy)
            self.axes.plot(xy[:, 0], xy[:, 1])

        return self.axes,


def main():
    hotel_data_dir = Path(Path(__file__).parent / 'data/datasets/eth/hotel')
    preprocessor = EthPreprosessor(hotel_data_dir, DatasetKind.hotel)
    df = preprocessor.preprocess_frame_data()

    animator = TrajectoryAnimator(df)
    ani = FuncAnimation(animator.fig, animator.update, interval=33)
    plt.show()


if __name__ == '__main__':
    main()
