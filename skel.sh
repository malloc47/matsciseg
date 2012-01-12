#!/bin/bash
export PYTHONPATH=./gco:/home/malloc47/usr/local/lib/python2.7/site-packages:/usr/lib64/python2.7/site-packages/mpich2
./matsciskel.py skel seq3/img/image0040.png seq3/labels/image0040.label seq3/img/image0041.png seq3/output2/image0041.label seq3/output2/image0041.png 10
for i in {42..50}; do
    k=$(($i - 1))
    ./matsciskel.py skel seq3/img/image00$k.png seq3/output2/image00$k.label seq3/img/image00$i.png seq3/output2/image00$i.label seq3/output2/image00$i.png 10
done