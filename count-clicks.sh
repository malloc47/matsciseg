#!/bin/bash
date > $2
script -c "xinput test $1" /dev/null | grep -v motion 1>> $2
date >> $2
dos2unix $2
