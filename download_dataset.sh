#!/usr/bin/env bash

# Example usage: ./download_dataset.sh 

# Arguments:
# -f : overwrites currently downloaded dataset
# -dataset: downloads the all datasets

# get options
while getopts ":fa" opt; do
  case $opt in
    f) force=true;;
    a) all=true;;
  esac
done
shift $((OPTIND -1))

URL=https://storage.googleapis.com/idp-chexpert-data/dev_dataset.zip
URL_FULL=https://storage.googleapis.com/idp-chexpert-data/CheXpert_small.zip
TMPFILE=`mktemp`
TMPFILE_FULL=`mktemp`
DEV_DATASET_DIRECTORY="$PWD/data/dev_dataset"
FULL_DATASET_DIRECTORY="$PWD/data/dataset"


if [[ -d "$DEV_DATASET_DIRECTORY" && ! $force ]] 
then
  echo "Dev dataset already exists, skipping download. Use -f to download anyways"
else
  # download and unzip dev dataset
  wget $URL -O $TMPFILE
  unzip $TMPFILE

  # move to data directory
  echo "Moving to data directory"
  mv ./home/jupyter/idp-radio-1/data/dev_dataset $DEV_DATASET_DIRECTORY

  # clean up
  echo "Cleaning up"
  rm -rf home
  rm $TMPFILE
fi

if [[ $all ]] 
then
  if [[ -d "$FULL_DATASET_DIRECTORY" && ! $force ]] 
  then
    echo "Dataset already exists, skipping download. Use -f to download anyways"
  else
    # download and unzip dev dataset
    wget $URL_FULL -O $TMPFILE_FULL
    unzip $TMPFILE_FULL

    # move to data directory
    echo "Moving to data directory"
    mkdir data/dataset
    mv "$PWD/CheXpert-v1.0-small" "$FULL_DATASET_DIRECTORY/CheXpert-v1.0-small"
    # move csv files to dataset directory
    mv "$FULL_DATASET_DIRECTORY/CheXpert-v1.0-small/train.csv" "$FULL_DATASET_DIRECTORY/train.csv"
    mv "$FULL_DATASET_DIRECTORY/CheXpert-v1.0-small/valid.csv" "$FULL_DATASET_DIRECTORY/valid.csv"

    # clean up
    echo "Cleaning up"
    rm $TMPFILE_FULL
  fi
fi

