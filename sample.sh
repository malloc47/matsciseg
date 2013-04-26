#!/bin/bash

# this command uses the "global" labeling recipe to transfer the
# segmentation of 0000.png to 0001.png; the -o switch additionally
# draws the edges on the image to visualize the output
./matscicli.py global -b e -d 10 sample/0000.png sample/0000.label sample/0001.png sample/0001.label -o sample/0001-output.png
