import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.01  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.05       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


# We have two inputs: inputs and distorted_inputs
# Distorted_inputs gives us the cutted and rotated picture
def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops. Used
    for training files

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)
    return images, labels


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops. This is usesd
    for creating testing files

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
      raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    return images, labels


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                         tf.nn.zero_fraction(x))


def fire(inputs,
         squeeze_depth,
         expand_depth,
         scope):
    """
    with tf.variable_scope(scope) as scope:
        # Get the number of input channel
        input_channel=int(inputs.get_shape()[3])
        # Squeeze part
        # kernel = tf.get_variable('weights', shape=[1, 1, input_channel, squeeze_depth],
        #                      initializer = tf.truncated_normal_initializer(stddev=5e-2))
        kernel = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, squeeze_depth], stddev=1e-2),
                               name='weights')
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
        # biases = tf.get_variable('biases', [squeeze_depth], initializer=tf.constant_initializer(0.0))
        biases = tf.Variable(tf.zeros(shape=[squeeze_depth]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        squeeze = tf.nn.relu(pre_activation, name='squeeze')

        # Expand part
        # kernel1x1 = tf.get_variable('weights1x1', shape=[1, 1, squeeze_depth, expand_depth],
        #                       initializer = tf.truncated_normal_initializer(stddev=5e-2))
        kernel1x1 = tf.Variable(tf.truncated_normal(shape=[1, 1, squeeze_depth, expand_depth], stddev=1e-2),
                                name='weights1x1')
        conv1x1 = tf.nn.conv2d(squeeze, kernel1x1, [1, 1, 1, 1], padding='SAME')
        # biases1x1 = tf.get_variable('biases1x1', [expand_depth], initializer=tf.constant_initializer(0.0))
        biases1x1 = tf.Variable(tf.zeros(shape=[expand_depth]), name='biases1x1')
        pre_activation1x1 = tf.nn.bias_add(conv1x1, biases1x1)
        expand1x1 = tf.nn.relu(pre_activation1x1)
        # kernel3x3 = tf.get_variable('weights3x3', shape=[3, 3, squeeze_depth, expand_depth],
        #                      initializer = tf.truncated_normal_initializer(stddev=5e-2))
        kernel3x3 = tf.Variable(tf.truncated_normal(shape=[3, 3, squeeze_depth, expand_depth], stddev=1e-2),
                                name='weights3x3')
        conv3x3 = tf.nn.conv2d(squeeze, kernel3x3, [1, 1, 1, 1], padding='SAME')
        # biases3x3 = tf.get_variable('biases3x3', [expand_depth], initializer=tf.constant_initializer(0.0))
        biases3x3 = tf.Variable(tf.zeros(shape=[expand_depth]), name='biases3x3')
        pre_activation3x3 = tf.nn.bias_add(conv3x3, biases3x3)
        expand3x3= tf.nn.relu(pre_activation3x3)
        expand=tf.concat([expand1x1, expand3x3], 3, name='expand')
    """
    squeeze = tf.layers.conv2d(inputs, squeeze_depth,  [1, 1], strides=1, name='squeeze_'+scope)
    _activation_summary(squeeze)
    e1x1 = tf.layers.conv2d(squeeze, expand_depth, [1, 1], strides=1, name='1x1_'+scope)
    e3x3 = tf.layers.conv2d(squeeze, expand_depth, [3, 3], padding="same", name='3x3_'+scope)
    expand = tf.concat([e1x1, e3x3], 3, name="expand_"+scope)
    _activation_summary(expand)

    return expand


# Modify this to achieve a squeezenet structure without using tf.layers
def inference(images, is_training=True):
    """Build the CIFAR-10 model.

    Args:
        images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #s

    with tf.variable_scope('squeezenet') as scope:
        # conv1
        with tf.variable_scope('conv1') as scope:
            # kernel = tf.get_variable('weights', shape=[2, 2, 3, 96],
            #                          initializer=tf.truncated_normal_initializer(stddev=5e-2))
            kernel = tf.Variable(tf.truncated_normal(shape=[2, 2, 3, 96], stddev=5e-2),
                               name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
            # biases = tf.get_variable('biases', [96], initializer=tf.constant_initializer(0.0))
            biases = tf.Variable(tf.zeros(shape=[96]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')

        # fire2
        fire2 = fire(pool1, squeeze_depth=16, expand_depth=64, scope='fire2')

        # fire3
        fire3 = fire(fire2, squeeze_depth=16, expand_depth=64, scope='fire3')

        # fire4
        fire4 = fire(fire3, squeeze_depth=32, expand_depth=128, scope='fire4')

        # pool4
        pool4= tf.nn.max_pool(fire4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='VALID', name='pool4')

        # fire5
        fire5 = fire(pool4, 32, 128, scope="fire5")

        # fire6
        fire6 = fire(fire5, 48, 192, scope="fire6")

        # fire7
        fire7 = fire(fire6, 48, 192, scope="fire7")

        # fire8
        fire8 = fire(fire7, 64, 256, scope="fire8")

        # pool8
        pool8 = tf.nn.max_pool(fire8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool8')

        # fire9
        fire9 = fire(pool8, 64, 256, scope="fire9")

        # drop9
        if is_training:
            drop9 = tf.nn.dropout(fire9, keep_prob=0.5, name="drop9")
        else:
            drop9 = tf.nn.dropout(fire9, keep_prob=1, name="drop9")

        # conv10
        with tf.variable_scope('conv10') as scope:
            # kernel = tf.get_variable('weights', shape=[1, 1, int(drop9.get_shape()[3]), NUM_CLASSES],
            #                          initializer=tf.truncated_normal_initializer(stddev=5e-2))
            kernel = tf.Variable(tf.truncated_normal(shape=[1, 1, int(drop9.get_shape()[3]), NUM_CLASSES], stddev=5e-2),
                                 name='weights')
            conv = tf.nn.conv2d(drop9, kernel, [1, 1, 1, 1], padding='VALID')
            # biases = tf.get_variable('biases', NUM_CLASSES), initializer=tf.constant_initializer(0.0))
            biases = tf.Variable(tf.zeros(shape=NUM_CLASSES), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv10 = tf.nn.relu(pre_activation, name=scope.name)

        # avgpool10
        avgpool10 = tf.nn.avg_pool(conv10, ksize=[1, 2, 2, 1],
                                   strides=[1, 1, 1, 1], padding="VALID")

        logits = tf.squeeze(avgpool10, [1, 2], name='logits')
        _activation_summary(logits)
    return logits


def loss(logits, labels):
    """ 
    Add L2loss to all the trainable variables 
    Add summary for "loss" and "loss/avg" 
    :param logits: logits from inference() 
    :param labels: labels from distorted_inputs or inputs() 1-D tensor of shape[batch_size] 
 
    :return: loss tensor of type float 
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                      global_step,
                                      decay_steps,
                                      LEARNING_RATE_DECAY_FACTOR,
                                      staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)


    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', total_loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(total_loss, global_step=global_step)
    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)