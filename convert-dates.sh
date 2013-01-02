#!/bin/bash
python -c "from dateutil import parser ; d1 = parser.parse(' $(head -1 $1 | tr -d '\n') ') ; d2 = parser.parse(' $(tail -1 $1 | tr -d '\n') ') ; print((d2-d1).seconds)" > ${1%.*}.time
