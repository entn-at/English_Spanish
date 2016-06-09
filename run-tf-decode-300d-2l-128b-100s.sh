#!/bin/bash -x

# Run tensorflow decoder
# with Spanish to English sequence to sequence translation  model

# model parameter sizes
# embedding dimension
embedsize=300
# number of cell layers
numlayers=2
batchsize=128
stepspercheckpoint=100

# directories
traindir=$(pwd)/train-${embedsize}-${numlayers}-${batchsize}-${stepspercheckpoint}
datadir=$(pwd)/input/all

# file extensions
srcx=es
tgtx=en

# data sizes
srcvocsize=$(find input/all/vocab*.${srcx} | ./prnum.pl)
tgtvocsize=$(find input/all/vocab*.${tgtx} | ./prnum.pl)

# run the decoder
python \
    ./decode.py \
    --decode True \
    --data_dir $datadir \
    --train_dir $traindir \
    --size $embedsize \
    --num_layers $numlayers \
    --src_extension $srcx \
    --tgt_extension $tgtx \
    --src_vocab_size $srcvocsize \
    --tgt_vocab_size $tgtvocsize \
    < input/test.$srcx
