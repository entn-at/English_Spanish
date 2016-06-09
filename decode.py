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


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    src_vocab_path = os.path.join(args.data_dir,
                                 "vocab%d." % args.src_vocab_size + args.src_extension)
    tgt_vocab_path = os.path.join(args.data_dir,
                                 "vocab%d." % args.tgt_vocab_size + args.tgt_extension)
    src_vocab, _ = data_tools.initialize_vocabulary(src_vocab_path)
    _, rev_tgt_vocab = data_tools.initialize_vocabulary(tgt_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_tools.sentence_to_token_ids(tf.compat.as_bytes(sentence), src_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_tools.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_tools.EOS_ID)]
      # Print out target sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_tgt_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def main(_):
    decode()

if __name__ == "__main__":
  args = parser.parse_args()
  tf.app.run()
