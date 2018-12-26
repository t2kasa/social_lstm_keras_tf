import tensorflow as tf

from social_lstm.compute_social_tensor import compute_social_tensor
from social_lstm.tfe_normal_sampler import normal2d_sample

_out_dim = 5


class SocialLSTM(tf.keras.Model):
    def __init__(self, pred_len, cell_side, n_side_cells, lstm_dim, emb_dim):
        super(SocialLSTM, self).__init__()
        self.pred_len = pred_len
        self.cell_side = cell_side
        self.n_side_cells = n_side_cells
        self.lstm_dim = lstm_dim
        self.emb_dim = emb_dim

        self.lstm_layer = tf.keras.layers.LSTM(lstm_dim, return_state=True)
        self.W_e_relu = tf.keras.layers.Dense(emb_dim, activation="relu")
        self.W_a_relu = tf.keras.layers.Dense(emb_dim, activation="relu")
        self.W_p = tf.keras.layers.Dense(_out_dim)

    def call(self, inputs, training=None, mask=None):
        # list of (pred_len, n_pids, out_dim)
        # note that n_pids may be different among inputs.
        o_pred_batch = [self._perform_sample(x) for x in inputs]
        return o_pred_batch

    def _perform_sample(self, x):
        """

        :param x: (obs_len, n_pids, xy_dim = 2)
        :return: output tensor. the shape is (pred_len, n_pids, out_dim)
        """
        obs_len, n_pids, _ = tf.shape(x).numpy()

        # ----------------------------------------------------------------------
        # observation step
        # ----------------------------------------------------------------------
        prev_h_t = tf.zeros((1, n_pids, self.lstm_dim))
        prev_c_t = tf.zeros((1, n_pids, self.lstm_dim))

        o_obs = []
        for t in range(obs_len):
            x_t = x[t, :, :]
            h_t, c_t, o_t = self._perform_step_t(x_t, prev_h_t, prev_c_t)

            o_obs.append(o_t)
            prev_h_t, prev_c_t = h_t, c_t
        # (batch_size = 1, obs_len, n_pids, out_dim = 5)
        o_obs = _stack_permute_axis_zero(o_obs)

        # ----------------------------------------------------------------------
        # prediction step
        # ----------------------------------------------------------------------
        # この時点でprev_h_t, prev_c_tにはobs_lenの最終的な状態が残っている

        # At the first prediction frame,
        # use the latest output of the observation step
        # (batch_size = 1, obs_len, n_pids, out_dim = 5)
        # => (n_pids, out_dim = 5)
        prev_o_t = o_obs[0, -1, :, :]

        o_pred = []
        for t in range(self.pred_len):
            # (n_pids, out_dim = 5) => (n_pids, 2)
            x_pred_t = normal2d_sample(prev_o_t)
            h_t, c_t, o_t = self._perform_step_t(x_pred_t, prev_h_t, prev_c_t)

            o_pred.append(o_t)
            prev_h_t, prev_c_t, prev_o_t = h_t, c_t, o_t

        # (pred_len, 1, n_pids, out_dim) => (1, pred_len, n_pids, out_dim)
        o_pred = _stack_permute_axis_zero(o_pred)
        return o_pred

    def _perform_step_t(self, x_t, prev_h_t, prev_c_t):
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
