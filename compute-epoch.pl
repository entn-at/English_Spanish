#!/usr/bin/perl -w
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 1 or croak "USAGE: compute_epoch.pl LOG_FILE";
}

my $batch_size = 256;
my $num_examples = 84224;
#open my $LOG, '<', "nohup.out" or croak "could not open file nohup.out for reading";
my @lines = ();

#LINE: while ( my $line = <$LOG> ) {
LINE: while ( my $line = <> ) {
    chomp $line;
    next LINE unless $line =~ /global step/;
    my @line = split /\s+/, $line, 10;
    push @lines, $line[2];
}
#close $LOG;

print $lines[$#lines] * $batch_size / $num_examples, "\n";
