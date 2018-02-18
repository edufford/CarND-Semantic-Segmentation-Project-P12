import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import time
import argparse
from moviepy.editor import VideoFileClip
import scipy.misc
import numpy as np
from PIL import Image

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
               'Please use TensorFlow version 1.0 or newer.  You are using {}' \
               .format(tf.__version__)

print('TensorFlow Version: {}'.format(tf.__version__))

# Suppress TensorFlow warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

  # Debug check VGG layer dimensions
  #print("vgg_layer3_out:", vgg_layer3_out.get_shape())
  #print("vgg_layer4_out:", vgg_layer4_out.get_shape())
  #print("vgg_layer7_out:", vgg_layer7_out.get_shape())

  # Implement FCN-8s architecture based on:
  # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
  #         voc-fcn8s-atonce/net.py

  # Kernel parameters
  kL2Reg = 0.001
  kInitSTD = 0.01

  # Apply 1x1 convolution to VGG layer 7 to reduce # of classes to num_classes
  score_fr = tf.layers.conv2d(vgg_layer7_out, num_classes,
            kernel_size=1, strides=1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
            kernel_initializer=tf.truncated_normal_initializer(stddev=kInitSTD),
            name='score_fr')

  # Upsample 2x by transposed convolution
  upscore2 = tf.layers.conv2d_transpose(score_fr, num_classes,
                    kernel_size=4, strides=2, padding='same',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
                    kernel_initializer=tf.zeros_initializer,
                    name='upscore2')

  # Rescale VGG layer 4 (max pool) for compatibility as a skip layer
  scale_pool4 = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

  # Apply 1x1 convolution to rescaled VGG layer 4 to reduce # of classes
  score_pool4 = tf.layers.conv2d(scale_pool4, num_classes,
            kernel_size=1, strides=1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
            kernel_initializer=tf.truncated_normal_initializer(stddev=kInitSTD),
            name='score_pool4')

  # Add skip layer from VGG layer 4
  fuse_pool4 = tf.add(upscore2, score_pool4)

  # Upsample 2x by transposed convolution
  upscore_pool4 = tf.layers.conv2d_transpose(fuse_pool4, num_classes,
                    kernel_size=4, strides=2, padding='same',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
                    kernel_initializer=tf.zeros_initializer,
                    name='upscore_pool4')

  # Rescale VGG layer 3 (max pool) for compatibility as a skip layer
  scale_pool3 = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')

  # Apply 1x1 convolution to rescaled VGG layer 3 to reduce # of classes
  score_pool3 = tf.layers.conv2d(scale_pool3, num_classes,
            kernel_size=1, strides=1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
            kernel_initializer=tf.truncated_normal_initializer(stddev=kInitSTD),
            name='score_pool3')

  # Add skip layer from VGG layer 3
  fuse_pool3 = tf.add(upscore_pool4, score_pool3)

  # Upsample 8x by transposed convolution
  upscore8 = tf.layers.conv2d_transpose(fuse_pool3, num_classes,
                    kernel_size=16, strides=8, padding='same',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
                    kernel_initializer=tf.zeros_initializer,
                    name='upscore8')

  # Add identity layer to name output
  fcn8s_out = tf.identity(upscore8, name='fcn8s_out')

  #tf.Print(fcn8_out, [tf.shape(fcn8_out)])

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

  # Add loss to TensorBoard summary logging
  tf.summary.scalar('loss', total_loss)

  # Set up Adam optimizer and training operation to minimize total loss
  global_step = tf.Variable(0, trainable=False, name='global_step')
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(total_loss, global_step=global_step,
                                name='train_op')

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

  # Set up TensorBoard logging output
  tb_out_dir = os.path.join('tb/', str(time.time()))
  tb_merged = tf.summary.merge_all()
  #train_writer = tf.summary.FileWriter(tb_out_dir, sess.graph) # with graph
  train_writer = tf.summary.FileWriter(tb_out_dir) # without graph

  # Initialize any uninitialized variables
  sess.run(tf.global_variables_initializer())

  # Train network
  for epoch in range(epochs):
    print("Epoch #", epoch+1)

    for image_batch, label_batch in get_batches_fn(batch_size):
      feed_dict = {input_image: image_batch,
                   correct_label: label_batch,
                   keep_prob: 0.5}

      # Run training step on each batch
      _, loss_value, summary = sess.run([train_op, cross_entropy_loss,
                                         tb_merged], feed_dict=feed_dict)

      # Log loss for each global step
      step = tf.train.global_step(sess, tf.train.get_global_step())
      train_writer.add_summary(summary, step)
      print("  Step", step, "loss =", loss_value)

print('\nTesting train_nn()...')
tests.test_train_nn(train_nn)


def parse_args():
  """
  Set up argument parser for command line operation of main.py program
  """

  parser = argparse.ArgumentParser()

  parser.add_argument('-md', '--mode',
    help='mode [1]: 0=Train, 1=Test, 2=Video', type=int, default=1)

  parser.add_argument('-ep', '--epochs',
    help='epochs [40]', type=int, default=40)

  parser.add_argument('-bs', '--batch_size',
    help='batch size [4]', type=int, default=4)

  parser.add_argument('-lr', '--learn_rate',
    help='learning rate [0.0001]', type=float, default=0.0001)

  args = parser.parse_args()
  return args


def run():
  """
  Run main semantic segmentation program
  """

  print('\nStarting run...')
  args = parse_args()

  # Basic parameters
  kNumClasses = 2 # "road" or "not road"
  kImageShape = (160, 576)
  data_dir = './data'
  runs_dir = './runs'
  model_path = './duffnet/'
  model_name = 'duffnet'

  # Hyperparameters
  epochs = args.epochs
  batch_size = args.batch_size
  learning_rate = args.learn_rate

  # TensorFlow placeholders
  correct_label = tf.placeholder(tf.bool, [None, None, None, kNumClasses])

  # Check data set validity
  print('\nTesting Kitti dataset...')
  tests.test_for_kitti_dataset(data_dir)

  # Download pretrained VGG model if necessary
  helper.maybe_download_pretrained_vgg(data_dir)

  # Path to VGG model
  vgg_path = os.path.join(data_dir, 'vgg')
  data_folder = os.path.join(data_dir, 'data_road/training')

  # Create generator function to get batches for training
  get_batches_fn = helper.gen_batch_function(data_folder, kImageShape)

  # Start TensorFlow session
  with tf.Session() as sess:

    ### Train new network ###
    if args.mode == 0:

      # Build NN using load_vgg, layers, and optimize function
      img_input, keep_prob, vgg3, vgg4, vgg7 = load_vgg(sess, vgg_path)
      fcn8s_out = layers(vgg3, vgg4, vgg7, kNumClasses)
      logits, train_op, loss = optimize(fcn8s_out, correct_label, learning_rate,
                                        kNumClasses)

      # Train NN using the train_nn function
      train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss,
               img_input, correct_label, keep_prob, learning_rate)

      # Save model result
      saver = tf.train.Saver()
      save_path = saver.save(sess, model_path+model_name)
      print("\nModel saved.")

    ### Test network ###
    elif args.mode == 1:

      # Load saved model
      saver = tf.train.import_meta_graph(model_path+model_name+'.meta')
      saver.restore(sess, tf.train.latest_checkpoint(model_path))
      graph = tf.get_default_graph()
      img_input = graph.get_tensor_by_name('image_input:0')
      keep_prob = graph.get_tensor_by_name('keep_prob:0')
      fcn8s_out = graph.get_tensor_by_name('fcn8s_out:0')
      logits = tf.reshape(fcn8s_out, (-1, kNumClasses))

      # Process test images
      helper.save_inference_samples(runs_dir, data_dir, sess, kImageShape,
                                    logits, keep_prob, img_input)

    ### Process video ###
    elif args.mode == 2:

      def process_frame(img):
        # Input image is a Numpy array, resize it to match NN input dimensions
        img_orig_size = (img.shape[0], img.shape[1])
        img_resized = scipy.misc.imresize(img, kImageShape)

        # Get NN tensors
        graph = tf.get_default_graph()
        img_input = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        fcn8s_out = graph.get_tensor_by_name('fcn8s_out:0')
        logits = tf.reshape(fcn8s_out, (-1, kNumClasses))

        # Process image with NN
        img_softmax = sess.run([tf.nn.softmax(logits)],
                               {keep_prob: 1.0, img_input: [img_resized]})

        # Reshape to 2D image dimensions
        img_softmax = img_softmax[0][:, 1].reshape(kImageShape[0],
                                                   kImageShape[1])

        # Threshold softmax probability to a binary road judgement (>50%)
        segmentation = (img_softmax > 0.5).reshape(kImageShape[0],
                                                   kImageShape[1], 1)

        # Apply road judgement to original image as a mask with alpha = 50%
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_img = Image.fromarray(img_resized)
        street_img.paste(mask, box=None, mask=mask)

        # Resize image back to original dimensions
        street_img_resized = scipy.misc.imresize(street_img, img_orig_size)

        # Output image as a Numpy array
        img_out = np.array(street_img_resized)
        return img_out

      # Load saved model
      saver = tf.train.import_meta_graph(model_path+model_name+'.meta')
      saver.restore(sess, tf.train.latest_checkpoint(model_path))

      # Process video frames
      video_outfile = './video/project_video_out.mp4'
      video = VideoFileClip('./video/project_video.mp4')#.subclip(37,38)
      video_out = video.fl_image(process_frame)
      video_out.write_videofile(video_outfile, audio=False)

    else:
      print('Error: Invalid mode selected.')


if __name__ == '__main__':
    run()
