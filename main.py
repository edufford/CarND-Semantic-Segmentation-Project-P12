import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import time
from tqdm import tqdm
import argparse

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
               'Please use TensorFlow version 1.0 or newer.  You are using {}' \
               .format(tf.__version__)

print('TensorFlow Version: {}'.format(tf.__version__))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check for a GPU
if not tf.test.gpu_device_name():
  warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
  print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
  """
  Load Pretrained VGG Model into TensorFlow.
  :param sess: TensorFlow Session
  :param vgg_path: Path to vgg folder with "variables/" and "saved_model.pb"
  :return: Tuple of Tensors from VGG model
           (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
  """
  vgg_tag = 'vgg16'
  vgg_input_tensor_name = 'image_input:0'
  vgg_keep_prob_tensor_name = 'keep_prob:0'
  vgg_layer3_out_tensor_name = 'layer3_out:0'
  vgg_layer4_out_tensor_name = 'layer4_out:0'
  vgg_layer7_out_tensor_name = 'layer7_out:0'

  tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
  graph = tf.get_default_graph()
  image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
  keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
  layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
  layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
  layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

  return image_input, keep_prob, layer3_out, layer4_out, layer7_out

print('\nTesting load_vgg()...')
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
  """
  Create the layers for a fully convolutional network.
  Build skip-layers using the vgg layers.
  :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
  :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
  :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
  :param num_classes: Number of classes to classify
  :return: The Tensor for the last layer of output
  """

  #print("vgg_layer3_out:", vgg_layer3_out.get_shape())
  #print("vgg_layer4_out:", vgg_layer4_out.get_shape())
  #print("vgg_layer7_out:", vgg_layer7_out.get_shape())

  # Implement FCN-8s architecture based on:
  # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s-atonce/net.py

  # Apply 1x1 convolution to VGG layer 7 to reduce # of classes to num_classes
  score_fr = tf.layers.conv2d(vgg_layer7_out, num_classes,
              kernel_size=1, strides=1, padding='same',
              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
              name='score_fr')

  # Upsample 2x by transposed convolution
  upscore2 = tf.layers.conv2d_transpose(score_fr, num_classes,
                    kernel_size=4, strides=2, padding='same',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                    kernel_initializer=tf.zeros_initializer,
                    name='upscore2')

  # Rescale VGG layer 4 (max pool) for compatibility as a skip layer
  scale_pool4 = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

  # Apply 1x1 convolution to rescaled VGG layer 4 to reduce # of classes
  score_pool4 = tf.layers.conv2d(scale_pool4, num_classes,
              kernel_size=1, strides=1, padding='same',
              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
              name='score_pool4')

  # Add skip layer from VGG layer 4
  fuse_pool4 = tf.add(upscore2, score_pool4)

  # Upsample 2x by transposed convolution
  upscore_pool4 = tf.layers.conv2d_transpose(fuse_pool4, num_classes,
                    kernel_size=4, strides=2, padding='same',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                    kernel_initializer=tf.zeros_initializer,
                    name='upscore_pool4')

  # Rescale VGG layer 3 (max pool) for compatibility as a skip layer
  scale_pool3 = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')

  # Apply 1x1 convolution to rescaled VGG layer 3 to reduce # of classes
  score_pool3 = tf.layers.conv2d(scale_pool3, num_classes,
              kernel_size=1, strides=1, padding='same',
              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
              name='score_pool3')

  # Add skip layer from VGG layer 3
  fuse_pool3 = tf.add(upscore_pool4, score_pool3)

  # Upsample 8x by transposed convolution
  upscore8 = tf.layers.conv2d_transpose(fuse_pool3, num_classes,
                    kernel_size=16, strides=8, padding='same',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                    kernel_initializer=tf.zeros_initializer,
                    name='upscore8')

  #tf.Print(fcn8_out, [tf.shape(fcn8_out)])

  fcn8s_out = tf.identity(upscore8, name='fcn8s_out')

  return fcn8s_out

print('\nTesting layers()...')
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
  """
  Build the TensorFLow loss and optimizer operations.
  :param nn_last_layer: TF Tensor of the last layer in the neural network
  :param correct_label: TF Placeholder for the correct label image
  :param learning_rate: TF Placeholder for the learning rate
  :param num_classes: Number of classes to classify
  :return: Tuple of (logits, train_op, total_loss)
  """

  # Reshape logits and labels
  logits = tf.reshape(nn_last_layer, (-1, num_classes))
  labels = tf.reshape(correct_label, (-1, num_classes))

  # Calculate softmax cross entropy loss and regularization loss operations
  cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                 logits=logits, labels=labels)
  cross_entropy_loss = tf.reduce_mean(cross_entropies)

  l2_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  regularization_loss = tf.reduce_sum(l2_reg_losses)

  total_loss = cross_entropy_loss + regularization_loss
  #total_loss_named = tf.identity(total_loss, name="total_loss")

  tf.summary.scalar('loss', total_loss)

  # Set up Adam optimizer and training operation to minimize total loss
  global_step = tf.Variable(0, trainable=False, name='global_step')
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(total_loss, global_step=global_step, name='train_op')

  return logits, train_op, total_loss

print('\nTesting optimize()...')
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image, correct_label, keep_prob,
             learning_rate):
  """
  Train neural network and print out the loss during training.
  :param sess: TF Session
  :param epochs: Number of epochs
  :param batch_size: Batch size
  :param get_batches_fn: Function to get batches of training data.
                         Call using get_batches_fn(batch_size)
  :param train_op: TF Operation to train the neural network
  :param cross_entropy_loss: TF Tensor for the amount of loss
  :param input_image: TF Placeholder for input images
  :param correct_label: TF Placeholder for label images
  :param keep_prob: TF Placeholder for dropout keep probability
  :param learning_rate: TF Placeholder for learning rate
  """

  # Initialize any uninitialized variables
  tb_out_dir = os.path.join('tb/', str(time.time()))
  tb_merged = tf.summary.merge_all()

  #train_writer = tf.summary.FileWriter(tb_out_dir, sess.graph) # with graph
  train_writer = tf.summary.FileWriter(tb_out_dir) # without graph

  sess.run(tf.global_variables_initializer())

  for epoch in range(epochs):
    print("Epoch #", epoch+1)

    for image_batch, label_batch in get_batches_fn(batch_size):
      feed_dict = {input_image: image_batch,
                   correct_label: label_batch,
                   keep_prob: 0.5}
      _, loss_value, summary = sess.run([train_op, cross_entropy_loss,
                                         tb_merged], feed_dict=feed_dict)

      step = tf.train.global_step(sess, tf.train.get_global_step())
      train_writer.add_summary(summary, step)
      print("  Step", step, "loss =", loss_value)

print('\nTesting train_nn()...')
tests.test_train_nn(train_nn)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-md', '--mode',
      help='mode: 0=train new, 1=train existing, 2=test. [0]', type=int, default=0)

    parser.add_argument('-ep', '--epochs',
      help='epochs [1]', type=int, default=1)

    parser.add_argument('-bs', '--batch_size',
      help='batch size [1]', type=int, default=1)

    parser.add_argument('-lr', '--learn_rate',
      help='learning rate [0.0001]', type=float, default=0.0001)

    #parser.add_argument('-tb', '--tensor_board', help='enable TensorBoard logging', action="store_true")
    args = parser.parse_args()
    return args

def run():

  print('\nStarting run...')
  args = parse_args()

  # Basic parameters
  kNumClasses = 2
  kImageShape = (160, 576)
  data_dir = './data'
  runs_dir = './runs'

  print('\nTesting Kitti dataset...')
  tests.test_for_kitti_dataset(data_dir) # check for Kitti data set

  # Hyperparameters
  epochs = args.epochs
  batch_size = args.batch_size
  learning_rate = args.learn_rate

  correct_label = tf.placeholder(tf.bool, [None, None, None, kNumClasses])

  # Download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)

  # OPTIONAL: Train and Inference on the cityscapes dataset instead of the
  #  Kitti dataset. You'll need a GPU with at least 10 teraFLOPS to train on.
  #  https://www.cityscapes-dataset.com/

  with tf.Session() as sess:

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    data_folder = os.path.join(data_dir, 'data_road/training')

    # Create generator function to get batches
    get_batches_fn = helper.gen_batch_function(data_folder, kImageShape)

    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network


    # Build NN using load_vgg, layers, and optimize function
    #if args.mode == 0: # Build new network
    img_input, keep_prob, vgg3, vgg4, vgg7 = load_vgg(sess, vgg_path)
    fcn8s_out = layers(vgg3, vgg4, vgg7, kNumClasses)
    logits, train_op, loss = optimize(fcn8s_out, correct_label, learning_rate,
                                        kNumClasses)
    """
    elif args.mode == 1: # Load existing network
      tf.saved_model.loader.load(sess, ['duffnet'], './duffnet')
      #saver = tf.train.import_meta_graph('./duffnet-2.meta')
      #saver.restore(sess, tf.train.latest_checkpoint('.'))

      graph = tf.get_default_graph()
      img_input = graph.get_tensor_by_name('image_input:0')
      keep_prob = graph.get_tensor_by_name('keep_prob:0')
      fcn8s_out = graph.get_tensor_by_name('fcn8s_out:0')
      logits = tf.reshape(fcn8s_out, (-1, kNumClasses))
      train_op = graph.get_tensor_by_name('train_op:0')
      loss = graph.get_tensor_by_name('total_loss:0')
"""
    # Train NN using the train_nn function
    train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss,
              img_input, correct_label, keep_prob, learning_rate)

    # Save model result
    builder = tf.saved_model.builder.SavedModelBuilder('./duffnet')
    builder.add_meta_graph_and_variables(sess, ['duffnet'])
    builder.save()

    #saver = tf.train.Saver()
    #step = tf.train.global_step(sess, tf.train.get_global_step())
    #saver.save(sess, './duffnet', global_step=step)
    print("\nModel saved.")

    # Save inference data using helper.save_inference_samples
    #if args.mode == 2:
    helper.save_inference_samples(runs_dir, data_dir, sess, kImageShape,
                                    logits, keep_prob, img_input)

    # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
