from argparse import Namespace

import numpy as np
import tensorflow as tf

from tfe_normal_sampler import normal2d_sample


def _stack_permute_axis_zero(xs):
    xs = tf.stack(xs, axis=0)
    perm = [1, 0] + list(range(2, xs.shape.ndims))
    xs = tf.transpose(xs, perm=perm)
    return xs


if __name__ == '__main__':
    tf.enable_eager_execution()
    args = Namespace(obs_len=3, pred_len=2, max_n_peds=52, pxy_dim=3,
                     grid_side=4, lstm_state_dim=32, emb_dim=16, out_dim=5)
    args.grid_side_squared = args.grid_side ** 2

    batch_size = 1
    x_input = np.random.randn(batch_size, args.obs_len, args.max_n_peds,
                              args.pxy_dim)
    x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
    grid_input = np.random.randn(batch_size, args.obs_len, args.max_n_peds,
                                 args.max_n_peds, args.grid_side_squared)
    grid_input = tf.convert_to_tensor(grid_input, dtype=tf.float32)

    # define layers
    lstm_layer = tf.keras.layers.LSTM(
        args.lstm_state_dim, return_state=True)
    W_e_relu = tf.keras.layers.Dense(args.emb_dim, activation="relu")
    W_a_relu = tf.keras.layers.Dense(args.emb_dim, activation="relu")
    W_p = tf.keras.layers.Dense(args.out_dim)

    # --------------------------------------------------------------------------
    # _build_model()
    # --------------------------------------------------------------------------

    prev_h_t = tf.zeros((1, args.max_n_peds, args.lstm_state_dim))
    prev_c_t = tf.zeros((1, args.max_n_peds, args.lstm_state_dim))

    o_obs_batch = []
    for t in range(args.obs_len):
        x_t = x_input[:, t, :, :]
        grid_t = grid_input[:, t, ...]
        h_t = []
        c_t = []
        o_t = []

        # social tensor
        H_t = np.random.randn(batch_size, args.max_n_peds,
                              (args.grid_side ** 2) * args.lstm_state_dim)
        H_t = tf.convert_to_tensor(H_t, dtype=tf.float32)

        pos_t = x_t[..., 1:]
        e_t = W_e_relu(pos_t)
        a_t = W_a_relu(H_t)
        emb_t = tf.concat([e_t, a_t], axis=-1)

        prev_states_t = [[prev_h_t[:, i], prev_c_t[:, i]] for i in
                         range(args.max_n_peds)]

        for ped_index in range(args.max_n_peds):
            # build concatenated embedding states as LSTM input
            emb_it = emb_t[:, ped_index, :]
            emb_it = tf.reshape(emb_it, (batch_size, 1, 2 * args.emb_dim))

            lstm_output, h_it, c_it = lstm_layer(
                emb_it, initial_state=prev_states_t[ped_index])

            h_t.append(h_it)
            c_t.append(c_it)

            # compute o_it, which shape is (b, 5)
            o_it = W_p(lstm_output)
            o_t.append(o_it)

        h_t = _stack_permute_axis_zero(h_t)
        c_t = _stack_permute_axis_zero(c_t)
        o_t = _stack_permute_axis_zero(o_t)

        o_obs_batch.append(o_t)

        prev_h_t, prev_c_t = h_t, c_t

    # (b, obs_len, max_n_peds, out_dim)
    o_obs_batch = _stack_permute_axis_zero(o_obs_batch)

    # --------------------------------------------------------------------------
    # prediction step
    # --------------------------------------------------------------------------
    # この時点でprev_h_t, prev_c_tにはobs_lenの最終的な状態が残っている

    # (b, obs_len, max_n_peds, pxy_dim) => (b, max_n_peds, pxy_dim)
    x_obs_t_final = x_input[:, -1, :, :]
    # (b, max_n_peds, pxy_dim) => (b, max_n_peds)
    pid_obs_t_final = x_obs_t_final[:, :, 0]
    # (b, max_n_peds) => (b, max_n_peds, 1)
    pid_obs_t_final = tf.expand_dims(pid_obs_t_final, axis=-1)

    x_pred_batch = []
    o_pred_batch = []

    # At the first prediction frame,
    # use the latest output of the observation step
    # (b, obs_len, max_n_peds, out_dim) => (b, max_n_peds, out_dim)
    prev_o_t = o_obs_batch[:, -1, :, :]

    for t in range(args.pred_len):

        # TODO: implement normal2d_sample() for eager mode
        pred_pos_t = normal2d_sample(prev_o_t)
        # assume all the pedestrians in the final observation frame are
        # exist in the prediction frames.

        x_pred_t = tf.concat([pid_obs_t_final, pred_pos_t], axis=2)

        # grid_t = tf_grid_mask(x_pred_t,
        #                       get_image_size(config.test_dataset_kind),
        #                       config.n_neighbor_pixels, config.grid_side)

        h_t, c_t, o_t = [], [], []

        # (n_samples, max_n_peds, (grid_side ** 2) * lstm_state_dim)
        # social tensor
        H_t = np.random.randn(batch_size, args.max_n_peds,
                              (args.grid_side ** 2) * args.lstm_state_dim)
        H_t = tf.convert_to_tensor(H_t, dtype=tf.float32)

        for i in range(args.max_n_peds):
            print("(t, li):", t, i)

            prev_o_it = prev_o_t[:, i, :]
            H_it = H_t[:, i, ...]

            # pred_pos_it: (batch_size, 2)
            pred_pos_it = normal2d_sample(prev_o_it)

            # e_it: (batch_size, emb_dim)
            # a_it: (batch_size, emb_dim)
            e_it = W_e_relu(pred_pos_it)
            a_it = W_a_relu(H_it)

            # build concatenated embedding states for LSTM input
            # emb_it: (batch_size, 1, 2 * emb_dim)
            emb_it = tf.concat([e_it, a_it], axis=1)
            emb_it = tf.reshape(emb_it, (batch_size, 1, 2 * args.emb_dim))

            # initial_state = h_i_tになっている
            # h_i_tを次のx_t_pに対してLSTMを適用するときのinitial_stateに使えば良い
            prev_states_it = [prev_h_t[:, i], prev_c_t[:, i]]
            lstm_output, h_it, c_it = lstm_layer(emb_it, prev_states_it)

            h_t.append(h_it)
            c_t.append(c_it)

            # compute output_it, which shape is (batch_size, 5)
            o_it = W_p(lstm_output)
            o_t.append(o_it)

        # convert lists of h_it/c_it/o_it to h_t/c_t/o_t respectively
        h_t = _stack_permute_axis_zero(h_t)
        c_t = _stack_permute_axis_zero(c_t)
        o_t = _stack_permute_axis_zero(o_t)

        o_pred_batch.append(o_t)
        x_pred_batch.append(x_pred_t)

        # current => previous
        prev_h_t = h_t
        prev_c_t = c_t
        prev_o_t = o_t

    # convert list of output_t to output_batch
    o_pred_batch = _stack_permute_axis_zero(o_pred_batch)
    x_pred_batch = _stack_permute_axis_zero(x_pred_batch)
    o_concat_batch = tf.concat([o_obs_batch, o_pred_batch], axis=1)

    lr = 0.003

    # # 本当に学習に必要なモデルはこっちのはず
    # self.train_model = Model([self.x_input, self.grid_input, self.zeros_input],
    #                          o_pred_batch)
    # optimizer = RMSprop(lr=lr)
    # self.train_model.compile(optimizer, self._compute_loss)
    # self.sample_model = Model([self.x_input, self.grid_input, self.zeros_input],
    #                           x_pred_batch)
