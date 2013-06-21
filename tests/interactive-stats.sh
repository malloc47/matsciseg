#!/bin/sh
for i in *.click ; do echo -n "${i%.*} " ; grep '^button press   1' $i | wc -l ; done > clicks
for i in *.click ; do ~/src/projects/matsci/matsciskel/convert-dates.sh $i ; done
for i in *.time ; do echo -n "${i%.*} " ; cat $i ; done > time
