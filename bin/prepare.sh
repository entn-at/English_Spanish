#!/bin/bash -x
#prepare.sh - run  corpus preparation scripts asuming train, test and tune folds have already been partitioned.
#tokenizes and normalizes data
#run from directory above bin directory
# assumes folds have already been separated

# directories
workingdir=`pwd`
bindir=$workingdir/bin
traindatadir=$workingdir/en-es
trainfilename=$traindatadir/UNv1.0.en-es
devdatadir=$workingdir/testsets/devset
devfilename=$devdatadir/UNv1.0.devset
testdatadir=$workingdir/testsets/testset
testfilename=$testdatadir/UNv1.0.testset
tokenizerdir=$workingdir/tokenizer
outdir=$workingdir/input/all
$(mkdir -p ${outdir})
# file extensions
inextension=en
outextension=es
# number of tokens output by cleaner 
min=1
max=38
# tools
tokenizer=$tokenizerdir/tokenizer.perl
cleaner=$bindir/clean-corpus-n.perl 
normalizer=$tokenizerdir/normalize-punctuation.perl
downcaser=$tokenizerdir/lowercase.perl

# tokenize and normalize
# test set
for ext in $inextension $outextension; do
    {
	while read line; do
	    $tokenizer | \
		$normalizer | \
		$downcaser
	done
    } > $outdir/test-dirty.${ext} < $testfilename.$ext
done

#dev set
for ext in $inextension $outextension; do
    {
	while read line; do
	    $tokenizer -l $ext | \
		$normalizer | \
		$downcaser
	done
    } > $outdir/dev-dirty.${ext} < $devfilename.$ext
done

#train set
for ext in $inextension $outextension; do
    {
	while read line; do
	    $tokenizer -l $ext | \
		$normalizer | \
		$downcaser
	done
    } > $outdir/train-dirty.${ext} < $trainfilename.$ext
done

for fld in test  train; do
    {
	$cleaner \
	    $outdir/${fld}-dirty \
	    $inextension \
	$outextension \
	    $outdir/$fld \
	    $min \
	    $max
    }

    rm $outdir/${fld}-dirty*
done

{
    $cleaner \
	$outdir/dev-dirty \
	$inextension \
	$outextension \
	$outdir/tune \
	$min \
	$max
}
rm $outdir/dev-dirty*

