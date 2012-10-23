#!/bin/bash
mkdir -p syn/$1/ground/
mkdir -p syn/$1/img/
./convert_vtk.py syn/$1.vtk syn/$1/ground/
for f in syn/$1/ground/* ; do
	noise/sample.py noise/data.pkl $f syn/$1/img/`basename $f .label`.png
done
