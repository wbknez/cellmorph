#!/usr/bin/env sh

# Removes all __pycache__ folders from every directory starting at the current
# one.

find . -type d -name "__pycache__" -print0 | while IIFS= read -r -d $'\0' dir; do
    rm -fr "${dir}"
    echo "Removed directory: ${dir}."
done
