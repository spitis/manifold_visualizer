import tensorflow as tf
import numpy as np
import os

from tensorflow.examples.tutorials.mnist import input_data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
tf.logging.set_verbosity(old_v)


def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()


def compute_accuracy(y, pred):
  correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
  return tf.reduce_mean(tf.cast(correct, tf.float32))


def build_autoencoder(x):

  # encoder
  h = tf.layers.dense(x, 1024, tf.nn.relu)
  h = tf.layers.dense(h, 128, tf.nn.relu)
  _z = tf.layers.dense(h, 16)
  z = tf.nn.tanh(_z)

  # decoder
  h = tf.layers.dense(z, 128, tf.nn.relu)
  h = tf.layers.dense(h, 1024, tf.nn.relu)
  x_proj = tf.layers.dense(h, 784, tf.nn.sigmoid)

  return z, _z, x_proj


def build_classifier(x):

  # 2-layer convolutional network
  h = tf.reshape(x, [-1, 28, 28, 1])
  h = tf.layers.conv2d(h, 32, 5, padding='same', activation=tf.nn.relu)
  h = tf.layers.max_pooling2d(h, 2, 2, 'SAME')
  h = tf.layers.conv2d(h, 64, 5, padding='same', activation=tf.nn.relu)
  h = tf.layers.max_pooling2d(h, 2, 2, 'SAME')
  h = tf.reshape(h, [-1, 7 * 7 * 64])
  h = tf.layers.dense(h, 1024, activation=tf.nn.relu)
  logits = tf.layers.dense(h, 10)

  return logits


def build_graph():
  reset_graph()

  x = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.float32, [None, 10])

  lr_autoencoder = tf.constant(1e-1)
  lr_classifier = tf.constant(1e-2)

  z, _z, x_proj = build_autoencoder(x)
  logits = build_classifier(x)

  loss_autoencoder = tf.reduce_mean(tf.squared_difference(
      x, x_proj)) + 1e-4 * tf.reduce_mean(tf.square(_z))
  loss_classifier = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

  preds = tf.nn.softmax(logits)
  accuracy = compute_accuracy(y, preds)

  ts_autoencoder = tf.train.AdamOptimizer(lr_autoencoder).minimize(
      loss_autoencoder)
  ts_classifer = tf.train.GradientDescentOptimizer(lr_classifier).minimize(
      loss_classifier)

  init_op = tf.global_variables_initializer()
  saver = tf.train.Saver()

  return locals()


class Model():

  def __init__(self):
    [setattr(self, k, v) for k, v in build_graph().items()]

    self.sess = sess = tf.Session()
    sess.run(self.init_op)

  def eval_autoencoder(self):
    return self.sess.run(
        self.loss_autoencoder,
        feed_dict={
            self.x: mnist.validation.images,
            self.y: mnist.validation.labels
        })

  def eval_classifier(self):
    return self.sess.run(
        [self.loss_autoencoder, self.accuracy],
        feed_dict={
            self.x: mnist.validation.images,
            self.y: mnist.validation.labels
        })

  def classifier_trainstep(self, batch_size):
    x, y = mnist.train.next_batch(batch_size)
    _ = self.sess.run(self.ts_classifer, {self.x: x, self.y: y})

  def train_autoencoder(self, num_epochs, batch_size=50, lr=1e-1):
    max_iters = int(mnist.train.num_examples / batch_size)

    for epoch in range(num_epochs):
      print("Epoch", epoch, " - Val loss:", self.eval_autoencoder())
      for i in range(max_iters):
        x, y = mnist.train.next_batch(batch_size)
        self.sess.run(
            self.ts_autoencoder,
            feed_dict={
                self.x: x,
                self.y: y,
                self.lr_autoencoder: lr
            })

    print("Epoch", num_epochs, " - Val loss:", self.eval_autoencoder())

  def save(self, target='saves/basic'):
    self.saver.save(self.sess, target)

  def load(self, target='saves/basic'):
    self.saver.restore(self.sess, './' + target)


class MnistManifoldGenerator():
  def __init__(self, g):
    """  
    g is a model object with:
    - `sess`: a Tensorflow session
    - `z`: a latent tensor
    - `x`: an input tensor (placeholder)
    - `x_proj`: tensor of the projection of the latent space back into the input space
    """
    self.g = g
  
  def __call__(self, length=100):
    g = self.g

    x = mnist.train.images[np.random.choice(range(mnist.train.num_examples))].reshape([1, -1])
    ims = [g.sess.run(g.z, {g.x: x})[0]]
    imset = []
    total_dist = 0

    while total_dist < length:
      if not imset:
        idxs = np.random.choice(range(mnist.train.num_examples), int(length / 2), replace=False)
        imgs = mnist.train.images[idxs]
        imgs_z = g.sess.run(g.z, feed_dict={g.x: imgs})

        imset = list(imgs_z)
        
      im = ims[-1]
      sqdists = np.sum(np.square(im - np.array(imset)), axis=1)
      min_idx = np.argmin(sqdists)
      dist = np.sqrt(sqdists[min_idx])
      total_dist += dist
      im2 = imset.pop(min_idx)
      vect = (im2 - im) / dist # normalized vector

      for i in range(1, int(dist / 0.02)):
        ims.append(im + vect*0.02*i)

      ims.append(im2)
      
    return g.sess.run(g.x_proj, feed_dict={g.z: np.array(ims[:length*30])})

if __name__ == "__main__":
  g = Model()
  if not os.path.exists('saves/basic.meta'):
    g.train_autoencoder(20, 20, 1e-3)
    g.save()
  g.load()
  res = g.eval_autoencoder()
  print("Autoencoder accuracy: {}".format(res))