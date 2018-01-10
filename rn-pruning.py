from datetime import datetime
import math
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import resnet as rn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('prune_dir', './cifar10_rneval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_rntrain',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('retrain_dir', './cifar10_rnretrain',
                           """Directory where to read model checkpoints for retrain.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 5,
                            """How often to run the eval.""")

# Modify this later
def regression_model(r, p):
    """Build the regression model, use tensorflow

    to see if there is an optimal reduce factor
    """
    # model parameters
    b = tf.Variable([1, 1, 10, 10, 40], dtype=tf.float32)
    # input and output
    x = tf.placeholder(tf.float32)
    logistic_model = b[0]-b[1]/(b[2]+b[3]*tf.exp(-b[4]*x))
    y = tf.placeholder(tf.float32)

    # loss and optimizer
    loss = tf.reduce_sum(tf.square(logistic_model-y))
    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(10000):
        sess.run(train, {x: r, y: p})

    # evaluate and plot
    b_value, loss_value, p_predict = sess.run([b, loss, logistic_model], {x: r, y: p})
    # p_predict2 = b_value[0]-b_value[1]/(b_value[2]+b_value[3]*tf.exp(-b_value[4]*r))
    print("loss is %s" % loss_value)

    plt.scatter(r, p, color='black')
    plt.plot(r, p_predict, color='blue', linewidth=3)
    plt.show()


def eval_once(sess, top_k_op, saver=None, global_step=None):
    """Run Eval once.
    """
    # Do evaluation
    num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * FLAGS.batch_size
    step = 0
    while step < num_iter:  # and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1
    # Compute precision @ 1.
    precision = true_count / total_sample_count
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    return precision


def prune():
    """Do pruning, and save pruned model for retrain
    """
    with tf.Graph().as_default() as g:
        # Input evaluation data
        images, labels = rn.inputs(eval_data=True)

        # inference model.
        logits = rn.inference(images, 15)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Create a saver
        saver = tf.train.Saver()

        # Create session to restore, and restore data
        sess = tf.InteractiveSession()

        # Queue runner
        tf.train.start_queue_runners()

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # extract global_step from it.
            global_step_num = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        precision = eval_once(sess, top_k_op)
        
    """
        # Get all variables
        lst_variables = tf.global_variables()
        lst_values = sess.run(tf.global_variables())

        # Get the pruning information
        r = np.arange(0,0.2,0.01)
        p = []
        for reduce_factor in r:
            kernel_index, channel_to_delete_pack, pruning_number_pack = \
                pru_cal(lst_variables, lst_values, reduce_factor=reduce_factor)
            print('reduce factor is %.3f' % reduce_factor)

            # Delete these variables
            counter = 0
            for i in kernel_index:
                for j in range(pruning_number_pack[counter]):
                    sess.run(tf.assign(lst_variables[i][:, :, :, channel_to_delete_pack[counter][j]],
                                       tf.zeros(
                                           tf.shape(lst_variables[i][:, :, :, channel_to_delete_pack[counter][j]])),
                                       name=lst_variables[i][:, :, :, channel_to_delete_pack[counter][j]].name))
                counter = counter + 1

            # Real evaluation, after pruning
            p.append(eval_once(sess, top_k_op))

        return r, p
    """


def pru_cal(variables, values, reduce_factor):
    """
    Calculate the index of the kernels we wanted to prune

    return: list of index in different layers, and the number
        of kernel to delete for each layer
    """
    names = [variable.name for variable in variables]
    values = np.array(values)
    values = [np.transpose(value) for value in values]

    kernel_index = []
    for name in names:
        if name.find("kernel") != -1 or name.find("weights") != -1:
            kernel_index.append(names.index(name))

    # The definition of redundancy
    channel_to_delete_pack = []
    pruning_number_pack = []
    for i in kernel_index:
        layer = values[i]
        M = np.sum(abs(layer)) / np.prod(np.shape(layer))
        channel = np.shape(layer)[0]
        S = np.zeros(channel)
        for j in range(channel):
            kernel = layer[j]
            s = np.sum(abs(kernel) < M) / np.prod(np.shape(kernel))
            S[j] = s
        index = np.argsort(S)
        channel_to_delete_pack.append(index)
        pruning_number = int(channel * reduce_factor)
        pruning_number_pack.append(pruning_number)

    return kernel_index, channel_to_delete_pack, pruning_number_pack


def main():
    prune()
    # r, p = prune()
    # regression_model(r, p)
    string = input("Press any key to exit.")


if __name__ == '__main__':
    main()
