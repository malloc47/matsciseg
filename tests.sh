#!/bin/bash
export PYTHONPATH=./gco:/home/malloc47/usr/local/lib/python2.7/site-packages:/usr/lib64/python2.7/site-packages/mpich2
./matsciskel.py skel seq3/img/image0040.png seq3/labels/image0040.label seq3/img/image0041.png test1.label test1.png 10
./matsciskel.py global seq1/img/image0090.tif seq1/labels/image0090.label seq1/img/image0091.tif test2.label test2.png 10
 ./matsciskel.py gauss seq5/img/image0001.png seq5/labels/image0001.label seq5/img/image0002.png seq5/labels/image0002.label seq5/output/image0002.png 40 5

 ./matsciskel.py gauss seq5/img/image0117.png seq5/labels/image0117.label seq5/img/image0118.png seq5/labels/image0118.label seq5/output/image0118.png 40 3
 ./matsciskel.py gauss seq5/img/image0118.png seq5/labels/image0118.label seq5/img/image0119.png seq5/labels/image0119.label seq5/output/image0119.png 40 3
 ./matsciskel.py gauss seq5/img/image0119.png seq5/labels/image0119.label seq5/img/image0120.png seq5/labels/image0120.label seq5/output/image0120.png 40 2

 ./matsciskel.py global seq5/img/image0117.png seq5/labels/image0117.label seq5/img/image0118.png seq5/labels/image0118c.label seq5/output/image0118c.png 40
 ./matsciskel.py global seq5/img/image0118.png seq5/labels/image0118c.label seq5/img/image0119.png seq5/labels/image0119c.label seq5/output/image0119c.png 40
 ./matsciskel.py global seq5/img/image0119.png seq5/labels/image0119c.label seq5/img/image0120.png seq5/labels/image0120c.label seq5/output/image0120c.png 40
