from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
  """
  Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. 
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 10000 == 0:
          print("  processing line %d" % counter)
        tokens = line.split()
        for w in tokens:
          if w in vocab:
            vocab[w] += 1
          else:
            vocab[w] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """
  Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary):
  """
  Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
  sentence: the sentence in bytes format to convert to token-ids.
  vocabulary: a dictionary mapping tokens to integers.

  Returns:
  a list of integers, the token-ids for the sentence.
  """
  words = sentence.split()
  return [vocabulary.get(w, UNK_ID) for w in words]

  
def data_to_token_ids(data_path, target_path, vocabulary_path):
  """
  turn data file into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
  data_path: path to the data file in one-sentence-per-line format.
  target_path: path where the file with token-ids will be created.
  vocabulary_path: path to the vocabulary file.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 10000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prep_data(data_dir, src_vocab_size, tgt_vocab_size, src_ext, tgt_ext):
  """
  create vocabularies .

  Args:
  data_dir: directory in which the data sets are stored.
  src_vocab_size: size of the source vocabulary to create and use.
  tgt_vocab_size: size of the target vocabulary to create and use.
  src_ext: extension for source language
  tgt_ext: extension for target language

  Returns:
  A tuple of 6 elements:
  (1) path to the token-ids for source training data-set,
  (2) path to the token-ids for target training data-set,
  (3) path to the token-ids for source development data-set,
  (4) path to the token-ids for target development data-set,
  (5) path to the source vocabulary file,
  (6) path to the target vocabulary file.
  """

  train_path = os.path.join(data_dir, 'train')
  dev_path = os.path.join(data_dir, 'tune')

  # Create vocabularies of the appropriate sizes.
  tgt_vocab_path = os.path.join(data_dir, "vocab%d." % tgt_vocab_size + tgt_ext )
  src_vocab_path = os.path.join(data_dir, "vocab%d."  % src_vocab_size + src_ext)
  create_vocabulary(tgt_vocab_path, train_path + '.' + tgt_ext, tgt_vocab_size)
  create_vocabulary(src_vocab_path, train_path + "." + src_ext, src_vocab_size)

  # Create token ids for the training data.
  tgt_train_ids_path = train_path + (".ids%d." % tgt_vocab_size + tgt_ext)
  src_train_ids_path = train_path + (".ids%d." % src_vocab_size + src_ext)
  data_to_token_ids(train_path + "." + tgt_ext, tgt_train_ids_path, tgt_vocab_path)
  data_to_token_ids(train_path + "." + src_ext, src_train_ids_path, src_vocab_path)

  # Create token ids for the development data.
  tgt_dev_ids_path = dev_path + (".ids%d." % tgt_vocab_size + tgt_ext)
  src_dev_ids_path = dev_path + (".ids%d." % src_vocab_size + src_ext)
  data_to_token_ids(dev_path + "." + tgt_ext, tgt_dev_ids_path, tgt_vocab_path)
  data_to_token_ids(dev_path + "." + src_ext, src_dev_ids_path, src_vocab_path)

  return (src_train_ids_path, tgt_train_ids_path,
          src_dev_ids_path, tgt_dev_ids_path,
          src_vocab_path, tgt_vocab_path)
