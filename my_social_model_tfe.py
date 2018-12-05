from argparse import Namespace

import numpy as np
import tensorflow as tf

from tfe_normal_sampler import normal2d_sample
from grid_tfe import compute_social_tensor


def _stack_permute_axis_zero(xs):
    xs = tf.stack(xs, axis=0)
    perm = [1, 0] + list(range(2, xs.shape.ndims))
    xs = tf.transpose(xs, perm=perm)
    return xs


def perform_step_t(x_t, prev_h_t, prev_c_t,
                   W_e_relu, W_a_relu, W_p, lstm_layer,
                   cell_side, n_side_cells):
    h_t, c_t, o_t = [], [], []

    # compute social tensor
    positions = x_t[0, :, 1:]
    hidden_states = prev_h_t[0]
    social_tensors_t = compute_social_tensor(positions, hidden_states,
                                             cell_side, n_side_cells)
    social_tensors_t = tf.expand_dims(social_tensors_t, axis=0)

    pos_t = x_t[..., 1:]
    e_t = W_e_relu(pos_t)
    a_t = W_a_relu(social_tensors_t)
    emb_t = tf.concat([e_t, a_t], axis=-1)
    prev_states_t = [[prev_h_t[:, i], prev_c_t[:, i]] for i in
                     range(args.max_n_peds)]

    for i in range(args.max_n_peds):
        # build concatenated embedding states as LSTM input
        emb_it = emb_t[:, i, :]
        emb_it = tf.reshape(emb_it, (batch_size, 1, 2 * args.emb_dim))

        lstm_output, h_it, c_it = lstm_layer(emb_it, prev_states_t[i])
        o_it = W_p(lstm_output)

        h_t.append(h_it)
        c_t.append(c_it)
        o_t.append(o_it)

    h_t, c_t, o_t = [_stack_permute_axis_zero(u) for u in [h_t, c_t, o_t]]
    return h_t, c_t, o_t


if __name__ == '__main__':
    tf.enable_eager_execution()
    args = Namespace(obs_len=3, pred_len=2, max_n_peds=52, pxy_dim=3,
                     cell_side=0.5, n_side_cells=4, n_states=32, emb_dim=16,
                     out_dim=5)
    args.n_side_cells_squared = args.n_side_cells ** 2

    # my implementation works only when the batch size equals to 1.
    batch_size = 1
    x_input = np.random.randn(batch_size, args.obs_len, args.max_n_peds,
                              args.pxy_dim)
    x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)

    # define layers
    lstm_layer = tf.keras.layers.LSTM(args.n_states, return_state=True)
    W_e_relu = tf.keras.layers.Dense(args.emb_dim, activation="relu")
    W_a_relu = tf.keras.layers.Dense(args.emb_dim, activation="relu")
    W_p = tf.keras.layers.Dense(args.out_dim)

    # --------------------------------------------------------------------------
    # observation step
    # --------------------------------------------------------------------------

    prev_h_t = tf.zeros((1, args.max_n_peds, args.n_states))
    prev_c_t = tf.zeros((1, args.max_n_peds, args.n_states))

    o_obs_batch = []
    for t in range(args.obs_len):
        x_t = x_input[:, t, :, :]
        h_t, c_t, o_t = perform_step_t(x_t, prev_h_t, prev_c_t, W_e_relu,
                                       W_a_relu, W_p, lstm_layer,
                                       args.cell_side, args.n_side_cells)
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

    x_pred_batch, o_pred_batch = [], []

    # At the first prediction frame,
    # use the latest output of the observation step
    # (b, obs_len, max_n_peds, out_dim) => (b, max_n_peds, out_dim)
    prev_o_t = o_obs_batch[:, -1, :, :]

    for t in range(args.pred_len):
        # assume all the pedestrians in the final observation frame are
        # exist in the prediction frames.
        pred_pos_t = normal2d_sample(prev_o_t)
        x_pred_t = tf.concat([pid_obs_t_final, pred_pos_t], axis=2)

        h_t, c_t, o_t = perform_step_t(x_pred_t, prev_h_t, prev_c_t, W_e_relu,
                                       W_a_relu, W_p, lstm_layer,
                                       args.cell_side, args.n_side_cells)
        x_pred_batch.append(x_pred_t)
        o_pred_batch.append(o_t)

        prev_h_t, prev_c_t, prev_o_t = h_t, c_t, o_t

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
