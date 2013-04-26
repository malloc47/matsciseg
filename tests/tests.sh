#!/bin/bash
#export PYTHONPATH=./gco:/home/malloc47/usr/local/lib/python2.7/site-packages:/usr/lib64/python2.7/site-packages/mpich2
./matscicli.py global -b e -d 10 seq1/img/image0090.png seq1/ground/image0090.label seq1/img/image0091.png test2.label -o test2.png
./matscicli.py skel -b i -d 10 seq3/img/image0040.png seq3/labels/image0040.label seq3/img/image0041.png test1.label -o test1.png
./matscicli.py gauss -b i -d 50 -d2 5 -d3 2 seq12/img/image0769.png seq12/ground/image0769.label seq12/img/image0770.png test3.label -o test3.png
