from argparse import Namespace

import numpy as np
import tensorflow as tf

from my_social_model_tfe import SocialLSTM

if __name__ == '__main__':
    tf.enable_eager_execution()
    args = Namespace(obs_len=3, pred_len=2, max_n_peds=52, pxy_dim=3,
                     cell_side=0.5, n_side_cells=4, lstm_dim=32, emb_dim=16,
                     out_dim=5)

    # my implementation works only when the batch size equals to 1.
    batch_size = 1
    x_input = np.random.randn(batch_size, args.obs_len, args.max_n_peds,
                              args.pxy_dim)
    x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)

    social_lstm = SocialLSTM(args.pred_len, args.cell_side, args.n_side_cells,
                             args.lstm_dim, args.emb_dim, args.out_dim)
    social_lstm.call(x_input)
