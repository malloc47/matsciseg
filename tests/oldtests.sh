#!/bin/bash
export PYTHONPATH=./gco:/home/malloc47/usr/local/lib/python2.7/site-packages:/usr/lib64/python2.7/site-packages/mpich2
./matsciskel.py i skel seq3/img/image0040.png seq3/labels/image0040.label seq3/img/image0041.png test1.label test1.png 10
./matsciskel.py e global seq1/img/image0090.png seq1/ground/image0090.label seq1/img/image0091.png test2.label test2.png 10
./matsciskel.py i gauss seq12/img/image0769.png seq12/ground/image0769.label seq12/img/image0770.png test3.label test3.png 50 5 2
# ./matsciskel.py i gauss seq5/img/image0001.png seq5/labels/image0001.label seq5/img/image0002.png seq5/labels/image0002.label seq5/output/image0002.png 40 5

# ./matsciskel.py i gauss seq5/img/image0117.png seq5/labels/image0117.label seq5/img/image0118.png seq5/labels/image0118.label seq5/output/image0118.png 40 3
# ./matsciskel.py i gauss seq5/img/image0118.png seq5/labels/image0118.label seq5/img/image0119.png seq5/labels/image0119.label seq5/output/image0119.png 40 3
# ./matsciskel.py i gauss seq5/img/image0119.png seq5/labels/image0119.label seq5/img/image0120.png seq5/labels/image0120.label seq5/output/image0120.png 40 2
#
# ./matsciskel.py i global seq5/img/image0117.png seq5/labels/image0117.label seq5/img/image0118.png seq5/labels/image0118c.label seq5/output/image0118c.png 40
# ./matsciskel.py i global seq5/img/image0118.png seq5/labels/image0118c.label seq5/img/image0119.png seq5/labels/image0119c.label seq5/output/image0119c.png 40
# ./matsciskel.py i global seq5/img/image0119.png seq5/labels/image0119c.label seq5/img/image0120.png seq5/labels/image0120c.label seq5/output/image0120c.png 40
#./matsciskel.py i gauss seq12/img/image0769.png seq5/ground/image0769.label seq12/img/image0770.png test3.label test3.png 50 5 2
./matsciskel.py i log seq12/img/image0769.png seq12/ground/image0769.label seq12/img/image0770.png test4.label test4.png 30 2 5
