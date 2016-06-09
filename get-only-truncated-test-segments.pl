#!/usr/bin/perl -w
# get -only-truncated-test-segments.pl - tensorflow crashes on segments with more than 38 tokens 

use strict;
use warnings;
use Carp;

my $srcx = "es";
my $tgtx = "en";

open my $SRC, '<', "input/all/test.$srcx" or croak "could not open file input/all/test.$srcx for reading $!";
open my $TGT, '<', "input/all/test.$tgtx" or croak "could not open file input/all/test.$tgtx for reading $!";

my @src = <$SRC>;
my @tgt = <$TGT>;

close $SRC;
close $TGT;

my @srcout = ();
my @tgtout = ();
my @line = ();

for my $line (0..$#src) {
    chomp $src[$line];
    chomp $tgt[$line];
    @line = split /\s+/, $src[$line];
    if ($#line < 39) {
	push @srcout, $src[$line];
	push @tgtout, $tgt[$line];
    } else {
	print "$#line\n";
    }
}

open my $SRCO, '+>', "input/test.$srcx" or croak "could not open file input/test.$srcx for writing $!";
open my $TGTO, '+>', "input/test.$tgtx" or croak "could not open file input/test.$tgtx for writing $!";

for my $l (0..$#srcout) {
    print $SRCO "$srcout[$l]\n";
    print $TGTO "$tgtout[$l]\n";
}

close $SRCO;
close $TGTO;
