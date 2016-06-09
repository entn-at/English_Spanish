#!/bin/bash -x

# train Spanish to English sequence to sequence models with tensorflow

# train and tune files are assumed to be under input/all
# assumes prepare-all.sh has been run first to get train tune and test folds
# the source (src) language is Spanish
# the target (tgt) language is English

# model size parameters
# vector dimension
embedsize=300
# number of lstm layers
numlayers=2
# training size parameters
batchsize=128
stepspercheckpoint=100

# directories
datadir=$(pwd)/input/all
traindir=$(pwd)/train-${embedsize}-${numlayers}-${batchsize}-${stepspercheckpoint}
mkdir -p $traindir

# file extensions
srcx=es
tgtx=en

# data sizes
srcvocsize=$(cat \
    $datadir/train.${srcx} \
    | sed -e s/\\s/\\n/g \
    | sort -u \
    | wc -l)

tgtvocsize=$(cat \
    $datadir/train.${tgtx} \
    | sed -e s/\\s/\\n/g \
    | sort -u \
    | wc -l)

# run the trainer 
python \
    ./train.py \
    --data_dir $datadir \
    --train_dir $traindir \
    --size $embedsize \
    --num_layers $numlayers \
    --src_extension $srcx \
    --tgt_extension $tgtx \
    --src_vocab_size $srcvocsize \
    --tgt_vocab_size $tgtvocsize \
    --steps_per_checkpoint $stepspercheckpoint \
    --batch_size $batchsize
 

