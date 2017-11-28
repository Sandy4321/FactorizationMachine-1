import tensorflow as tf
from argparse import Namespace


class Model(object):
    def __init__(self, config: Namespace):
        self.batch_size = config.batch_size
        self.feature_size = config.feature_size
        self.factor_dim = config.factor_dim
        self.use_cross_entropy = config.use_cross_entropy
        self.learning_rate = config.learning_rate

        # define input, output and parameters
        self.x = tf.placeholder('float', shape=[self.batch_size, self.feature_size])
        self.y = tf.placeholder('float', shape=[self.batch_size, 1])

        self._w_0 = tf.Variable(tf.zeros([1]))
        self._w = tf.Variable(tf.zeros([self.feature_size]))

        self._V = tf.Variable(
            tf.random_normal([self.factor_dim, self.feature_size], stddev=0.1))
        self._loss = None
        self._optimizer = None

    def build_model(self):
        linear_terms = tf.add(self._w_0,         # linear part, w0+wx
                              tf.reduce_sum(
                                  tf.multiply(self._w, self.x), 1, keep_dims=True
                              ))

        interactions = tf.multiply(0.5,          # feature cross part
                                   tf.reduce_sum(
                                       tf.subtract(
                                           tf.pow(tf.matmul(self.x, tf.transpose(self._V)), 2),
                                           tf.matmul(tf.pow(self.x, 2), tf.transpose(tf.pow(self._V, 2)))
                                       ),
                                       1, keep_dims=True
                                   ))

        y_hat = tf.add(linear_terms, interactions)

        lambda_w = tf.constant(0.001, name='lambda_w')
        lambda_v = tf.constant(0.001, name='lambda_v')

        l2_norm = (tf.reduce_sum(               # L2 normalization for avoiding overfitting
                    tf.add(
                        tf.multiply(lambda_w, tf.pow(self._w, 2)),
                        tf.multiply(lambda_v, tf.pow(self._V, 2)))))

        if self.use_cross_entropy:
            error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=y_hat))
        else:
            error = tf.multiply(0.5, tf.reduce_mean(tf.square(tf.subtract(self.y, y_hat))))

        self._loss = tf.add(error, l2_norm)
        self._optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self._loss)

    def get_optimizer(self) -> tf.train.Optimizer:
        return self._optimizer

    def get_loss_var(self):
        return self._loss

    def get_w(self, sess: tf.Session):
        return sess.run(self._w)

    def get_v(self, sess: tf.Session):
        return sess.run(self._V).T
