import tensorflow as tf
import importlib
import tensorflow.python.platform
import os
import numpy as np
from datetime import datetime
from tensorflow.python.platform import gfile
from data import *
from evaluate import evaluate
import tensorflow.contrib.slim as slim


timestr = '-'.join(str(x) for x in list(tuple(datetime.now().timetuple())[:6]))
MOVING_AVERAGE_DECAY = 0.997
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 30,
                            """Number of epochs to train. -1 for unlimited""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            """Initial learning rate used.""")
tf.app.flags.DEFINE_string('model', 'model',
                           """Name of loaded model.""")
tf.app.flags.DEFINE_string('save', timestr,
                           """Name of saved dir.""")
tf.app.flags.DEFINE_string('load', None,
                           """Name of loaded dir.""")
tf.app.flags.DEFINE_string('dataset', 'cifar10',
                           """Name of dataset used.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './results/',
                           """results folder.""")
tf.app.flags.DEFINE_string('log_dir', './results',
                           """log folder.""")
tf.app.flags.DEFINE_bool('gpu', True,
                           """use gpu.""")
tf.app.flags.DEFINE_integer('device', 0,
                           """which gpu to use.""")
tf.app.flags.DEFINE_bool('summary', True,
                           """Record summary.""")
tf.app.flags.DEFINE_string('log', 'ERROR',
                           'The threshold for what messages will be logged '
                            """DEBUG, INFO, WARN, ERROR, or FATAL.""")


FLAGS.checkpoint_dir +=FLAGS.save
FLAGS.log_dir += '/log/'
# tf.logging.set_verbosity(FLAGS.log)

trainable_variables = ['alexnet_v2/conv1/weights:0',
                       'alexnet_v2/conv1/biases:0',

                       'alexnet_v2/conv2/weights:0',
                       'alexnet_v2/conv2/biases:0',

                       'alexnet_v2/conv3/weights:0',
                       'alexnet_v2/conv3/biases:0',

                       'alexnet_v2/conv4/weights:0',
                       'alexnet_v2/conv4/biases:0',

                       'alexnet_v2/conv5/weights:0',
                       'alexnet_v2/conv5/biases:0',

                       'alexnet_v2/fc6/weights:0',
                       'alexnet_v2/fc6/biases:0',

                       'alexnet_v2/fc7/weights:0',
                       'alexnet_v2/fc7/biases:0',

                       'alexnet_v2/fc8/weights:0',
                       'alexnet_v2/fc8/biases:0']


# trainable_variables = ['vgg_16/conv1/conv1_1/weights:0',
#                        'vgg_16/conv1/conv1_1/biases:0',
#                        'vgg_16/conv1/conv1_2/weights:0',
#                        'vgg_16/conv1/conv1_2/biases:0',
#
#                        'vgg_16/conv2/conv2_1/weights:0',
#                        'vgg_16/conv2/conv2_1/biases:0',
#                        'vgg_16/conv2/conv2_2/weights:0',
#                        'vgg_16/conv2/conv2_2/biases:0',
#
#                        'vgg_16/conv3/conv3_1/weights:0',
#                        'vgg_16/conv3/conv3_1/biases:0',
#                        'vgg_16/conv3/conv3_2/weights:0',
#                        'vgg_16/conv3/conv3_2/biases:0',
#                        'vgg_16/conv3/conv3_3/weights:0',
#                        'vgg_16/conv3/conv3_3/biases:0',
#
#                        'vgg_16/conv4/conv4_1/weights:0',
#                        'vgg_16/conv4/conv4_1/biases:0',
#                        'vgg_16/conv4/conv4_2/weights:0',
#                        'vgg_16/conv4/conv4_2/biases:0',
#                        'vgg_16/conv4/conv4_3/weights:0',
#                        'vgg_16/conv4/conv4_3/biases:0',
#
#                        'vgg_16/conv5/conv5_1/weights:0',
#                        'vgg_16/conv5/conv5_1/biases:0',
#                        'vgg_16/conv5/conv5_2/weights:0',
#                        'vgg_16/conv5/conv5_2/biases:0',
#                        'vgg_16/conv5/conv5_3/weights:0',
#                        'vgg_16/conv5/conv5_3/biases:0',
#
#                        'vgg_16/fc6/weights:0',
#                        'vgg_16/fc6/biases:0',
#
#                        'vgg_16/fc7/weights:0',
#                        'vgg_16/fc7/biases:0',
#
#                        'vgg_16/fc8/weights:0',
#                        'vgg_16/fc8/biases:0']


def count_params(var_list):
    num = 0
    for var in var_list:
        if var is not None:
            num += var.get_shape().num_elements()
    return num


def add_summaries(scalar_list=[], activation_list=[], var_list=[], grad_list=[]):

    for var in scalar_list:
        if var is not None:
            tf.summary.scalar(var.op.name, var)

    for grad, var in grad_list:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    for var in var_list:
        if var is not None:
            tf.summary.histogram(var.op.name, var)
            sz = var.get_shape().as_list()
            if len(sz) == 4 and sz[2] == 3:
                kernels = tf.transpose(var, [3, 0, 1, 2])
                tf.summary.image(var.op.name + '/kernels',
                                 group_batch_images(kernels), max_outputs=1)
    for activation in activation_list:
        if activation is not None:
            tf.summary.histogram(activation.op.name +
                                 '/activations', activation)


def _learning_rate_decay_fn(learning_rate, global_step):
  return tf.train.exponential_decay(
      learning_rate,
      global_step,
      decay_steps=20000,
      decay_rate=0.5,
      staircase=True)


learning_rate_decay_fn = _learning_rate_decay_fn


def train(model, data,
          batch_size=128,
          learning_rate=FLAGS.learning_rate,
          log_dir='./log',
          checkpoint_dir='./checkpoint',
          num_epochs=-1):

    # tf Graph input
    with tf.device('/cpu:0'):
        with tf.name_scope('data'):
            x, yt = data.generate_batches(batch_size)
        global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64,
                             initializer=tf.constant_initializer(0),
                             trainable=False)
    if FLAGS.gpu:
        device_str='/gpu:' + str(FLAGS.device)
    else:
        device_str='/cpu:0'
    with tf.device(device_str):
        alpha = tf.Variable(initial_value=0.0, trainable=False)
        y = model(x, alpha, is_training=True)
        # Define loss and optimizer
        with tf.name_scope('objective'):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
            # tf.add_to_collection("losses", loss)
            # loss = tf.add_n(tf.get_collection("losses"))
            accuracy = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))

        opt = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, 'Adam',
                                              gradient_noise_scale=None, gradient_multipliers=None,
                                              clip_gradients=None,  # moving_average_decay=0.9,
                                              learning_rate_decay_fn=learning_rate_decay_fn, update_ops=None, variables=None,
                                              name=None)
# grads = opt.compute_gradients(loss)
# apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # alpha
    lamda_start = 0.5
    lamda_fin = 1.0
    lamda_decay = (lamda_fin/lamda_start)**(1./num_epochs)

    # loss_avg

    ema = tf.train.ExponentialMovingAverage(
       MOVING_AVERAGE_DECAY, global_step, name='average')
    ema_op = ema.apply([loss, accuracy] + tf.trainable_variables())
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    loss_avg = ema.average(loss)
    tf.summary.scalar('loss/training', loss_avg)
    accuracy_avg = ema.average(accuracy)
    tf.summary.scalar('accuracy/training', accuracy_avg)

    check_loss = tf.check_numerics(loss, 'model diverged: loss->nan')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, check_loss)
    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([opt]):
        train_op = tf.group(*updates_collection)

    if FLAGS.summary:
        add_summaries(scalar_list=[accuracy, accuracy_avg, loss, loss_avg],
                      activation_list=tf.get_collection(tf.GraphKeys.ACTIVATIONS),
                      var_list=tf.trainable_variables())
        # grad_list=grads)
    summary_op = tf.summary.merge_all()

    # Configure options for session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(
        config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=gpu_options,
        )
    )

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # var_list = [var for var in tf.global_variables() if "moving" in var.name]
    # var_list += tf.trainable_variables()
    # saver = tf.train.Saver(var_list=var_list)

    # saver = tf.train.Saver()
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #     # Restores from checkpoint
    #    saver.restore(sess, ckpt.model_checkpoint_path)
    # else:
    #    print('No checkpoint file found')

    variables = slim.get_variables_to_restore(include=trainable_variables)
    print [v.name for v in variables]
    saver = tf.train.Saver(variables)
    saver_all = tf.train.Saver()
    saver.restore(sess, './alexnet_pretrained_model_withBN/vgg16.ckpt')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    epoch = 0
    lamda = lamda_start
    print('num of trainable paramaters: %d' %
          count_params(tf.trainable_variables()))
    saver_all.save(sess, save_path=checkpoint_dir +
                               '/vgg16.ckpt')
    while epoch != num_epochs:
        lamda = lamda * lamda_decay
        epoch += 1
        curr_step = 0
        # Initializing the variables

        print('Started epoch %d' % epoch)
        sess.run(tf.assign(alpha, lamda))
        while curr_step < data.size[0]:
            _, loss_val = sess.run([train_op, loss])
            curr_step += FLAGS.batch_size

        step, acc_value, loss_value, summary = sess.run(
            [global_step, accuracy_avg, loss_avg, summary_op])
        saver_all.save(sess, save_path='./results/vgg16.ckpt')
        print('alpha: %f' % alpha.eval(session=sess))
        print('Finished epoch %d' % epoch)
        print('Training Accuracy: %.3f' % acc_value)
        print('Training Loss: %.3f' % loss_value)

        test_acc_top1, test_acc_top5, test_loss = evaluate(model, FLAGS.dataset,
                                   batch_size=batch_size,
                                   checkpoint_dir=checkpoint_dir)

        print('Test Accuracy_Top1: %.3f' % test_acc_top1)
        print('Test Accuracy_Top5: %.3f' % test_acc_top5)
        print('Test Loss: %.3f' % test_loss)

        summary_out = tf.Summary()
        summary_out.ParseFromString(summary)
        summary_out.value.add(tag='accuracy_top1/test', simple_value=test_acc_top1)
        summary_out.value.add(tag='accuracy_top5/test', simple_value=test_acc_top5)
        summary_out.value.add(tag='loss/test', simple_value=test_loss)
        summary_writer.add_summary(summary_out, step)
        summary_writer.flush()

    # When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    summary_writer.close()


def main(argv=None):  # pylint: disable=unused-argument

    m = importlib.import_module('models.' +FLAGS.model)
    data = get_data_provider(FLAGS.dataset, training=True)

    train(m.model, data,
          batch_size=FLAGS.batch_size,
          checkpoint_dir=FLAGS.checkpoint_dir,
          log_dir=FLAGS.log_dir,
          num_epochs=FLAGS.num_epochs)


if __name__ == '__main__':
    tf.app.run()
