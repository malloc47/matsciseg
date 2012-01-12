#!/bin/bash
export PYTHONPATH=./gco:/home/malloc47/usr/local/lib/python2.7/site-packages:/usr/lib64/python2.7/site-packages/mpich2
./matsciskel.py skel seq3/img/image0040.png seq3/labels/image0040.label seq3/img/image0041.png test1.label test1.png 10
./matsciskel.py global seq1/img/image0090.tif seq1/labels/image0090.label seq1/img/image0091.tif test2.label test2.png 10