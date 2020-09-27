#!/bin/sh

ALL_FILES=$(find . -path "*ipynb_checkpoints*" -prune -false -o -name '*.ipynb')
# echo $ALL_FILES

for ipynb in $ALL_FILES
do
    jupyter nbconvert --to markdown $ipynb
done
