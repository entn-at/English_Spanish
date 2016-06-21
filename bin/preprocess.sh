#!/bin/bash -x
#  preprocesses a sample corpus,
# including tokenization, truecasing, and subword segmentation. 
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,
# and the file names 

# in the tokenization step, remove Romanian-specific normalization / diacritic removal,
# and  add your own.
# also, learn BPE segmentations separately for each language,
# especially if they differ in their alphabet
min=1
max=80
# suffix of source language files
SRC=es
# suffix of target language files
TRG=en
workingdir=$(pwd)
datadir=$workingdir/input/all
bindir=$workingdir/bin
modeldir=$workingdir/nematus-model
mkdir -p $modeldir $datadir

# number of merge operations.
# Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=89500
tooldir=/home/jjm/tools
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$tooldir/mosesdecoder
# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$tooldir/subword-nmt
# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/jjm/tools/nematus
# tokenize
for fld in test tune train; do
    for ext in $SRC $TRG; do
	cat $datadir/$fld.$ext \
	    | \
	    $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl \
		-l $ext \
	    | \
	    $mosesdecoder/scripts/tokenizer/tokenizer.perl \
		-a \
		-l $ext \
		> \
		$datadir/$fld.tok.$ext
    done

    # clean empty and long sentences,
    #and sentences with high source-target ratio
    $mosesdecoder/scripts/training/clean-corpus-n.perl \
	$datadir/$fld.tok \
	$SRC \
	$TRG \
	$datadir/$fld.tok.clean \
	$min \
	$max

    # train truecaser
    $mosesdecoder/scripts/recaser/train-truecaser.perl \
	-corpus $datadir/$fld.tok.clean.$SRC \
	-model $modeldir/truecase-model.$SRC

    $mosesdecoder/scripts/recaser/train-truecaser.perl \
	-corpus $datadir/$fld.tok.clean.$TRG \
	-model $modeldir/truecase-model.$TRG

    # apply truecaser 
    $mosesdecoder/scripts/recaser/truecase.perl \
	-model $modeldir/truecase-model.$SRC \
	< \
	$datadir/$fld.tok.clean.$SRC \
	> \
	$datadir/$fld.tc.$SRC

    $mosesdecoder/scripts/recaser/truecase.perl \
	-model $modeldir/truecase-model.$TRG \
	< \
	$datadir/$fld.tok.clean.$TRG \
	> \
	$datadir/$fld.tc.$TRG

    # train BPE
    cat $datadir/$fld.tc.$SRC \
	$datadir/$fld.tc.$TRG \
	| \
	$subword_nmt/learn_bpe.py \
	    -s $bpe_operations \
	    > \
	    $modeldir/$SRC$TRG.bpe

    # apply BPE
    $subword_nmt/apply_bpe.py \
	-c $modeldir/$SRC$TRG.bpe \
	< \
	$datadir/$fld.tc.$SRC \
	> \
	$datadir/$fld.bpe.$SRC

    $subword_nmt/apply_bpe.py \
	-c $modeldir/$SRC$TRG.bpe \
	< \
	$datadir/$fld.tc.$TRG \
	> \
	$datadir/$fld.bpe.$TRG

    # build network dictionary
    $nematus/data/build_dictionary.py \
 	$datadir/$fld.bpe.$SRC \
	$datadir/$fld.bpe.$TRG
done
