import numpy
import os
import sys

VOCAB_SIZE = 90000
SRC = "es"
TGT = "en"
DATA_DIR = "input/all/"

from nematus.nmt import train


if __name__ == '__main__':
    validerr = train(
        alpha_c=0.,  # alignment regularization
        batch_size=80,
        clip_c=1.,
        datasets=[DATA_DIR + '/train.bpe.' + SRC, DATA_DIR + '/train.bpe.' + TGT],
        decay_c=0.,
        decoder='gru_cond',
        dictionaries=[DATA_DIR + '/train.bpe.' + SRC + '.json',DATA_DIR + '/train.bpe.' + TGT + '.json'],
        dim=1024,
        dim_per_factor=None, 
        dim_word=500,
        dispFreq=1000,
        dropout_embedding=0.2, # dropout for input embeddings 0: no dropout
        dropout_hidden=0.2, # dropout for hidden layers 0: no dropout
        dropout_source=0.1, # dropout source words 0: no dropout
        dropout_target=0.1, # dropout target words 0: no dropout
        encoder='gru',
        external_validation_script='./validate.sh',
        factors=1,
        finetune=False,
        finish_after=10000000,  # finish after this many updates
        lrate=0.0001,
        max_epochs=5000,
        maxibatch_size=20,
        maxlen=50,
        n_words=VOCAB_SIZE,
        n_words_src=VOCAB_SIZE,
        optimizer='adadelta',
        overwrite=False,
        reload_=True,
        sampleFreq=1000,
        saveFreq=1000,
        saveto='nematus-model/model.npz',
        use_dropout=False,
        validFreq=1000,
        valid_batch_size=80,
        valid_datasets=[DATA_DIR + '/tune.bpe.' + SRC, DATA_DIR + '/tune.bpe.' + TGT]
    )
    print validerr
