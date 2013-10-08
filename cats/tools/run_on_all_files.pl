#!/usr/bin/perl
use strict;
use warnings;
use File::Find;
use File::Spec;
use Cwd;

my $directory = $ARGV[0];
my $script = $ARGV[1];
my $numargs = $#ARGV + 1;

$script = File::Spec->rel2abs( $script );

if($numargs != 2) {
    print_usage();
    die("ERROR: Wrong number of arguments\n");
}

sub print_usage {
    print("Usage: ");
    print($0);
    print(" DIRECTORY SCRIPT\n");
    print("Tool to run a python script recursively on all sub files in the directory specified\n\n");
}

unless(-e $script) {
    print_usage();
    die("ERROR: Second argument does not excist!\n");
}

unless(-e $directory) {
    print_usage();
    die("ERROR: First argument does not excist!\n");
}

unless(-d $directory) {
    print_usage();
    die("ERROR: First argument is not an directory!\n");
}

if($script !~ /..\.py/) {
    print_usage();
    die("ERROR: Second argument is not a python script!\n");
}



my @files = ();

find(\&wanted,$directory);

sub wanted {
    if ($File::Find::name =~ m/..\.lp/) {
	push(@files,[File::Spec->rel2abs($_),(stat($_))[7]]);
	#system("/home/kim/cass/mylpex2.py",getcwd."/" .$_ . "\n","o");	   
	#print getcwd."/" .$_ . "\n";
    }
}

map {
    print("File: $$_[0]\n");
    die() if($$_[0] =~ m/..2000../);
    system($script,$$_[0],"o");
#    die();
    
} sort {
    $$a[1] <=> $$b[1];
} @files;

