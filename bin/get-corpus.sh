#!/bin/bash
SRC=es
TRG=en
tar -zxvf /mnt/corpora/UN-Parallel-Corpus/Plain-text-bitexts/UNv1.0.en-es.tar.gz
tar -zxvf /mnt/corpora/UN-Parallel-Corpus/Test-and-Development-Sets/UNv1.0.testsets.tar.gz

workingdir=$(pwd)
datadir=$workingdir/input/all
mkdir -p $datadir
traindatadir=$workingdir/en-es
trainfilename=$traindatadir/UNv1.0.en-es
devdatadir=$workingdir/testsets/devset
devfilename=$devdatadir/UNv1.0.devset
testdatadir=$workingdir/testsets/testset
testfilename=$testdatadir/UNv1.0.testset
for ext in $SRC $TRG; do
    mv $trainfilename.$ext $datadir/train.$ext
    mv $devfilename.$ext $datadir/tune.$ext
    mv $testfilename.$ext $datadir/test.$ext
done

