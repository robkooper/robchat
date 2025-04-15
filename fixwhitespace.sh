#!/bin/bash

for x in app.py tests/*.py example/*.py; do
  sed -i~ 's/ *$//' $x
  if [ "$(tail -c1 $x | wc -l)" -eq 0 ]; then
    echo "" >> $x
  fi
done
