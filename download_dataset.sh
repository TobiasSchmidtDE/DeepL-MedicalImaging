#!/usr/bin/env bash

# Example usage: ./download_dataset.sh 

# Arguments:
# -f : overwrites currently downloaded dataset

# get options
while getopts ":Mmp" opt; do
  case $opt in
    f) force=true;;
  esac
done
shift $((OPTIND -1))


URL=https://storage.googleapis.com/idp-chexpert-data/dev_dataset.zip
TMPFILE=`mktemp`
DEV_DATASET_DIRECTORY="$PWD/data/dev_dataset"

if [[ -d "$DEV_DATASET_DIRECTORY" && !$force ]] 
then
  echo "Dataset already exists, skipping download. Use -f to download anyways"
else
  # download and unzip dev dataset
  wget $URL -O $TMPFILE
  unzip $TMPFILE

  # move to data directory
  echo "Moving to data directory"
  mv ./home/jupyter/idp-radio-1/data ./data

  # clean up
  echo "Cleaning up"
  rm -rf home
  rm $TMPFILE
fi
