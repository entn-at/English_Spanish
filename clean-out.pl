#!/usr/bin/perl -w
# clean-out.pl - get only lines output by the decoder 

use strict;
use warnings;
use Carp;;

BEGIN {
    @ARGV == 1 or croak "USAGE: clean-out.pl LOGFILE";
}

LINE: while ( my $line = <> ) {
    chomp $line;
    # only accept lines starting with >
    next LINE unless ($line =~ /^\>/);
    next LINE if ($line =~ /^>\sTraceback/);
    # remove the >
    $line =~ s/^\>//;
    print "$line\n";
}

