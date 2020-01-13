from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from data import get_data_provider


def evaluate(model, dataset,
        batch_size=128,
        checkpoint_dir='./checkpoint'):
    with tf.Graph().as_default() as g:
        data = get_data_provider(dataset, training=False)
        with tf.device('/cpu:1'):
            x, yt = data.generate_batches(batch_size)

        # Build the Graph that computes the logits predictions
        alpha = tf.Variable(initial_value=0.0)
        y = model(x, alpha, is_training=False)

        # Calculate predictions.
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
        accuracy_top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))
        accuracy_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y, yt, 5), tf.float32))

        # var_list = [var for var in tf.global_variables() if "moving" in var.name]
        # var_list += tf.trainable_variables()
        # saver = tf.train.Saver(var_list=var_list)

        # Configure options for session
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(
                config=tf.ConfigProto(
                            log_device_placement=False,
                            allow_soft_placement=True,
                            gpu_options=gpu_options,
                            )
                        )
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        saver.restore(sess, save_path='./results/vgg16.ckpt')

         # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
             threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))
            num_batches = int(math.ceil(data.size[0] / batch_size))
            total_acc_top1 = 0  # Counts the number of correct predictions per batch.
            total_acc_top5 = 0
            total_loss = 0  # Sum the loss of predictions per batch.
            step = 0
            #sess.run(tf.assign(alpha, 1.0))
            #bar = Bar('Evaluating', max=num_batches,suffix='%(percent)d%% eta: %(eta)ds')
            while step < num_batches and not coord.should_stop():
                a = datetime.now()
                acc_val_top1, acc_val_top5, loss_val = sess.run([accuracy_top1, accuracy_top5, loss])
                b = datetime.now()
                total_acc_top1 += acc_val_top1
                total_acc_top5 += acc_val_top5
                total_loss += loss_val
                step += 1

            # Compute precision and loss
            total_acc_top1 /= num_batches
            total_acc_top5 /= num_batches
            total_loss /= num_batches

            #bar.finish()
            time = (b - a).total_seconds()
            print('Calculate time: %.5f' % time)
            #print('alpha: %f' % alpha.eval(session=sess))

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads)
        return total_acc_top1, total_acc_top5, total_loss


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('checkpoint_dir', './results',
                             """Directory where to read model checkpoints.""")
  tf.app.flags.DEFINE_string('dataset', 'cifar10',
                             """Name of dataset used.""")
  tf.app.flags.DEFINE_string('model', 'model',
                             """Name of loaded model.""")

  FLAGS.log_dir = FLAGS.checkpoint_dir+'/log/'

  tf.app.run()
