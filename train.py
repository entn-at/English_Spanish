from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser, RawTextHelpFormatter
import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_tools
from tensorflow.models.rnn.translate import seq2seq_model


parser = ArgumentParser(
  description="""
  Binary for training translation models and decoding from them.

  Running this program without --decode will start training a model saving
  checkpoints to --train_dir.

  Running with --decode starts an interactive loop so you can see how
  the current checkpoint translates source sentences .
  """, formatter_class=RawTextHelpFormatter)

parser.add_argument(
  "-d",
  "--data_dir",
  type=str,
  help='Directory for storing data',
  default='input')

parser.add_argument(
  "-o",
  "--train_dir",
  type=str,
  help='Directory for storing training',
  default='train')

parser.add_argument(
  "-e",
  "--src_vocab_size",
  type=int,
  help="maximum number  of words in source vocabulary.",
  default=10000)

parser.add_argument(
  "-f",
  "--tgt_vocab_size",
  type=int,
  help="maximum number  of words in target vocabulary.",
  default=10000)

parser.add_argument(
  "-s",
  "--steps_per_checkpoint",
  type=int,
  help="Number of steps between check points.",
  default=50)

parser.add_argument(
  "-l",
  "--num_layers",
  type=int,
  help="the number of LSTM layers",
  default=2)

parser.add_argument(
  "-i",
  "--size",
  type=int,
  help="Size of each model layer.",
  default=128)

parser.add_argument(
  "-g",
  "--max_gradient_norm",
  type=float,
  help="the maximum permissible norm of the gradient",
  default=5.0)

parser.add_argument(
  "-b",
  "--batch_size",
  type=int,
  help="the batch size",
  default=64)

parser.add_argument(
  "-c",
  "--learning_rate_decay_factor",
  type=float,
  help="the decay of the learning rate for each epoch after max_epoch",
  default=0.99)

parser.add_argument(
  "-r",
  "--learning_rate",
  type=float,
  help="the initial value of the learning rate",
  default=0.5)

parser.add_argument(
  "-t",
  "--max_train_data_size",
  type=int,
  help="Limit on the size of training data (0: no limit).",
  default=0)

parser.add_argument(
  "-a",
  "--decode",
  type=bool,
  help="Set to True for interactive decoding.",
  default=False)

parser.add_argument(
  "-j",
  "--src_extension",
  type=str,
  help='Extension for source language.',
  default='en')

parser.add_argument(
  "-k",
  "--tgt_extension",
  type=str,
  help='Extension for target language.',
  default='ps')

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  """
  Read data from source and target files and put into buckets.

  Args:
  source_path: path to the files with token-ids for the source language.
  target_path: path to the file with token-ids for the target language;
  it must be aligned with the source file: n-th line contains the desired
  output for n-th line from the source_path.
  max_size: maximum number of lines to read, all other will be ignored;
  if 0 or None, data files will be read completely (no limit).

  Returns:
  data_set: a list of length len(_buckets); data_set[n] contains a list of
  (source, target) pairs read from the provided data files that fit
  into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
  len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_tools.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      args.src_vocab_size, args.tgt_vocab_size, _buckets,
      args.size, args.num_layers, args.max_gradient_norm, args.batch_size,
      args.learning_rate, args.learning_rate_decay_factor,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(args.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  # Prepare  data.
  print("Preparing  data in %s" % args.data_dir)
  src_train, tgt_train, src_dev, tgt_dev, _, _ = data_tools.prep_data(
    args.data_dir, args.src_vocab_size, args.tgt_vocab_size, args.src_extension,
    args.tgt_extension)


  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (args.num_layers, args.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % args.max_train_data_size)
    dev_set = read_data(src_dev, tgt_dev)
    train_set = read_data(src_train, tgt_train, args.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / args.steps_per_checkpoint
      loss += step_loss / args.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % args.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(args.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def main(_):
  train()

if __name__ == "__main__":
  args = parser.parse_args()
  tf.app.run()
