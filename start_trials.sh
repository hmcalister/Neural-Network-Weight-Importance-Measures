#!/bin/bash

usage() { echo "USAGE: $0 -n <int> -o <int>" 1>&2; exit 1; }

num_threads=0;
thread_offset=0;

while getopts "n:o:" flag
do
    case "${flag}" in 
        n) num_threads=${OPTARG};;
		o) thread_offset=${OPTARG};;
    esac
done

if [ $num_threads -le 0 ] 
then
    echo "PLEASE ASSIGN NUM THREADS TO AT LEAST 1";
    exit 1;
fi

for (( i=$thread_offset; i<$num_threads; i++ ))
do  
    python3 trial_script.py $i
done
