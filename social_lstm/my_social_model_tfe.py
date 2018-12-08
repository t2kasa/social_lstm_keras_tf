import tensorflow as tf

from social_lstm.grid_tfe import compute_social_tensor
from social_lstm.tfe_normal_sampler import normal2d_sample


class SocialLSTM(tf.keras.Model):
    def __init__(self, pred_len, cell_side, n_side_cells, lstm_dim, emb_dim,
                 out_dim=5):
        super(SocialLSTM, self).__init__()
        self.pred_len = pred_len
        self.cell_side = cell_side
        self.n_side_cells = n_side_cells
        self.lstm_dim = lstm_dim
        self.emb_dim = emb_dim

        self.lstm_layer = tf.keras.layers.LSTM(lstm_dim, return_state=True)
        self.W_e_relu = tf.keras.layers.Dense(emb_dim, activation="relu")
        self.W_a_relu = tf.keras.layers.Dense(emb_dim, activation="relu")
        self.W_p = tf.keras.layers.Dense(out_dim)

    def call(self, inputs, training=None, mask=None):
        for x in inputs:
            self.perform_sample(x)

    def perform_sample(self, x):
        obs_len, n_pids, _ = tf.shape(x).numpy()
        # ----------------------------------------------------------------------
        # observation step
        # ----------------------------------------------------------------------
        prev_h_t = tf.zeros((1, n_pids, self.lstm_dim))
        prev_c_t = tf.zeros((1, n_pids, self.lstm_dim))

        o_obs_batch = []
        for t in range(obs_len):
            x_t = x[t, :, :]
            h_t, c_t, o_t = self.perform_step_t(x_t, prev_h_t, prev_c_t)
            o_obs_batch.append(o_t)
            prev_h_t, prev_c_t = h_t, c_t

        # (b, obs_len, max_n_peds, out_dim)
        o_obs_batch = _stack_permute_axis_zero(o_obs_batch)

        # ----------------------------------------------------------------------
        # prediction step
        # ----------------------------------------------------------------------
        # この時点でprev_h_t, prev_c_tにはobs_lenの最終的な状態が残っている

        # (obs_len, n_pids, 2) => (n_pids, 2)
        x_obs_t_final = x[-1, :, :]
        # (b, n_pids, 2) => (b, n_pids)
        pid_obs_t_final = x_obs_t_final[:, :, 0]
        # (b, max_n_peds) => (b, max_n_peds, 1)
        pid_obs_t_final = tf.expand_dims(pid_obs_t_final, axis=-1)

        x_pred_batch, o_pred_batch = [], []

        # At the first prediction frame,
        # use the latest output of the observation step
        # (b, obs_len, max_n_peds, out_dim) => (b, max_n_peds, out_dim)
        prev_o_t = o_obs_batch[:, -1, :, :]

        for t in range(self.pred_len):
            # assume all the pedestrians in the final observation frame are
            # exist in the prediction frames.
            pred_pos_t = normal2d_sample(prev_o_t)
            x_pred_t = tf.concat([pid_obs_t_final, pred_pos_t], axis=2)

            h_t, c_t, o_t = self.perform_step_t(x_pred_t, prev_h_t, prev_c_t)
            x_pred_batch.append(x_pred_t)
            o_pred_batch.append(o_t)

            prev_h_t, prev_c_t, prev_o_t = h_t, c_t, o_t

        o_pred_batch = _stack_permute_axis_zero(o_pred_batch)
        return o_pred_batch

    def perform_step_t(self, x_t, prev_h_t, prev_c_t):
        n_pids, _ = tf.shape(x_t).numpy()
        h_t, c_t, o_t = [], [], []

        # compute social tensor
        hidden_states = prev_h_t[0]
        social_tensors_t = compute_social_tensor(
            x_t, hidden_states, self.cell_side, self.n_side_cells)
        # social_tensors_t = tf.expand_dims(social_tensors_t, axis=0)

        # (n_pids, 2) => (n_pids, emb_dim)
        e_t = self.W_e_relu(x_t)
        # (n_pids, 2) => (n_pids, emb_dim)
        a_t = self.W_a_relu(social_tensors_t)
        # [(n_pids, emb_dim), (n_pids, emb_dim)] => (n_pids, 2 * emb_dim)
        emb_t = tf.concat([e_t, a_t], axis=-1)
        prev_states_t = [[prev_h_t[:, i], prev_c_t[:, i]] for i in
                         range(n_pids)]

        for i in range(n_pids):
            # (n_pids, 2 * emb_dim) => (2 * emb_dim,)
            emb_it = emb_t[i, :]
            # (2 * emb_dim,) => (batch_size = 1, time_steps = 1, 2 * emb_dim)
            emb_it = tf.reshape(emb_it, (1, 1, 2 * self.emb_dim))

            lstm_output, h_it, c_it = self.lstm_layer(emb_it, prev_states_t[i])
            o_it = self.W_p(lstm_output)

            h_t.append(h_it)
            c_t.append(c_it)
            o_t.append(o_it)

        h_t, c_t, o_t = [_stack_permute_axis_zero(u) for u in [h_t, c_t, o_t]]
        return h_t, c_t, o_t


def _stack_permute_axis_zero(xs):
    xs = tf.stack(xs, axis=0)
    perm = [1, 0] + list(range(2, xs.shape.ndims))
    xs = tf.transpose(xs, perm=perm)
    return xs
