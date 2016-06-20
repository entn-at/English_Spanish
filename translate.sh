#!/bin/sh

# theano device
device=gpu

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/jjm/tools/nematus
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m nematus-model/model.npz \
     -i input/all/test.bpe.es \
     -o input/all/test.output \
     -k 12 \
     -n \
     -p 1
