#!/usr/bin/perl -w

while (<>) {
    chomp;
    $n = /(\d+)/g;
    if ($n) {
	print "$1\n";
    }
}

