import tensorflow as tf
import resnet as rn
import time
import numpy as np
import os
from datetime import datetime
from six.moves import xrange
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './cifar10_rntrain',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def run_training():
    """Training process"""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = rn.distorted_inputs()
            # images_eval, labels_eval = rn.inputs(eval_data=True)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = rn.inference(images, 15)

        # Calculate loss.
        loss = rn.loss(logits, labels)
        rn._activation_summary(loss)

        # The precision
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = rn.training(loss, global_step)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(tf.global_variables())

        # Create a session for running Ops on the Graph.
        sess = tf.InteractiveSession()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # And then after everything is built:
        # Run the Op to initialize the variables.
        sess.run(init)

        # Start all threads
        tf.train.start_queue_runners()

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            image_batch, label_batch = sess.run([images, labels])

            # This is the real training step
            _, loss_value = sess.run([train_op, loss])

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 10 == 0:
                # Print status to stdout.
                print('%s: Step %d: loss = %.2f (%.3f sec)' % 
                      (datetime.now(), step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            """
            if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
                with tf.device('/cpu:0'):
                    print(" Step %d: precision = %.2f" % (step,
                          sess.run(tf.reduce_sum(tf.to_int32([top_k_op])))/FLAGS.batch_size))

            """

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print(" Step %d, checkpoint saved! " % (step))


def main():
    '''
    By modifying the code below we solve the problem of
    unable to overlap
    '''
    rn.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    else:
        tf.gfile.MakeDirs(FLAGS.train_dir)
    run_training()
    str=input("Press any key to continue.")

if __name__ == '__main__':
    main()
