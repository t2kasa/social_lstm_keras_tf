from pathlib import Path

import tensorflow as tf

from social_lstm.losses import compute_loss
from social_lstm.metrics import compute_abe_tf
from social_lstm.tfe_normal_sampler import normal2d_sample


class Trainer:
    def __init__(self, model, optimizer, train_ds, test_ds, n_epochs, out_dir):
        self.model = model
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.n_epochs = n_epochs

        # create summaries
        train_dir = Path(out_dir, 'train').as_posix()
        test_dir = Path(out_dir, 'test').as_posix()
        self.train_summary_writer = tf.contrib.summary.create_file_writer(
            train_dir, name='train')
        self.test_summary_writer = tf.contrib.summary.create_file_writer(
            test_dir, name='test')

        # create checkpoint
        model_dir = Path(out_dir, 'checkpoints/').as_posix()
        self.checkpoint_prefix = Path(model_dir, 'ckpt').as_posix()
        self.step_counter = tf.train.get_or_create_global_step()
        self.checkpoint = tf.train.Checkpoint(
            model=model, optimizer=optimizer, step_counter=self.step_counter)
        self.checkpoint.restore(tf.train.latest_checkpoint(model_dir))

        Path(out_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        with tf.device('/gpu:0'):
            for epoch in range(self.n_epochs):
                print(f'epoch: {epoch:03d}')
                with self.train_summary_writer.as_default():
                    self.train_epoch()
                with self.test_summary_writer.as_default():
                    self.eval_epoch()

                self.checkpoint.save(self.checkpoint_prefix)

    def train_epoch(self, log_interval=5):
        for index, (pos_obs_true, pos_future_true) in enumerate(self.train_ds):
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                    log_interval, global_step=self.step_counter):
                with tf.GradientTape() as tape:
                    o_future_pred = self.model(pos_obs_true)
                    loss_value = compute_loss(pos_future_true, o_future_pred)

                pos_future_true = tf.reshape(pos_future_true, [-1, 2])
                pos_future_pred = normal2d_sample(o_future_pred)
                abe = compute_abe_tf(pos_future_true, pos_future_pred)

                tf.contrib.summary.scalar('abe', abe)
                tf.contrib.summary.scalar('loss', loss_value)
                # compute grads and update model params
                grads = tape.gradient(loss_value, self.model.variables)
                self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                               global_step=self.step_counter)

                if self.step_counter.numpy() % log_interval == 0:
                    print(f'step: {index + 1:04d}\t'
                          f'loss: {loss_value}\t'
                          f'abe: {abe}')

    def eval_epoch(self):
        loss_mean = tf.contrib.eager.metrics.Mean('loss', tf.float32)
        abe_mean = tf.contrib.eager.metrics.Mean('abe', tf.float32)

        for pos_obs_true, pos_future_true in self.test_ds:
            o_future_pred = self.model(pos_obs_true)
            loss_mean(compute_loss(pos_future_true, o_future_pred))

            pos_future_true = tf.reshape(pos_future_true, [-1, 2])
            pos_future_pred = normal2d_sample(o_future_pred)
            abe_mean(compute_abe_tf(pos_future_true, pos_future_pred))

        print(f'test loss: {loss_mean.result()}, abe: {abe_mean.result()}')
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', loss_mean.result())
            tf.contrib.summary.scalar('abe', abe_mean.result())
