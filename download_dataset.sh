#!/usr/bin/env bash
# Author: Tobias

# Downloads smallest version of Chexpert and Chestxray14 dataset by default. Optionally downloads all versions.

# Example usage: ./download_dataset.sh 

# Arguments:
# -f: overwrites currently downloaded dataset
# -s: downloads the server datasets (chexpert_full and chestxray14_256)
# -a: downloads the all datasets

# get options
while getopts ":fas" opt; do
  case $opt in
    f) force=true;;
    a) all=true;;
    s) server=true;;
  esac
done
shift $((OPTIND -1))

URL_CHEXPERT_DEV=https://storage.googleapis.com/idp-datasets/chexpert/chexpert_dev_medium_scale.tar.gz
URL_CHEXPERT_FULL=https://storage.googleapis.com/idp-datasets/chexpert/chexpert_full_medium_scale.tar.gz
URL_CHESTXRAY14_256=https://storage.googleapis.com/idp-datasets/chestxray14-nih/chestxray14_256.tar.gz
URL_CHESTXRAY14_512=https://storage.googleapis.com/idp-datasets/chestxray14-nih/chestxray14_512.tar.gz

CHEXPERT_DEV_DATASET_DIRECTORY="$PWD/data/chexpert/dev/"
CHEXPERT_FULL_DATASET_DIRECTORY="$PWD/data/chexpert/full/"
CHESTXRAY14_256_DATASET_DIRECTORY="$PWD/data/chestxray14/scale_256/"
CHESTXRAY14_512_DATASET_DIRECTORY="$PWD/data/chestxray14/scale_512/"



if [[ -d "$CHEXPERT_DEV_DATASET_DIRECTORY" && ! $force ]] 
then
  echo "Chexpert dev dataset already exists, skipping download. Use -f to download anyways"
else
  ### CHEXPERT_DEV_DATASET ###
  # download and unzip dev dataset
  echo "Downloading Chexpert dev dataset..."
  wget --no-check-certificate $URL_CHEXPERT_DEV -O "tmp.tar.gz"
  mkdir -p tmp
  tar -zxf "tmp.tar.gz" --directory tmp
  
  # move to data directory
  echo "Moving to data directory"
  rm -r -f $CHEXPERT_DEV_DATASET_DIRECTORY
  mkdir -p $CHEXPERT_DEV_DATASET_DIRECTORY
  mv tmp/*  $CHEXPERT_DEV_DATASET_DIRECTORY

  # clean up
  echo "Cleaning up"
  rm "tmp.tar.gz"
  rm -rf tmp/
fi

if [[ -d "$CHESTXRAY14_256_DATASET_DIRECTORY" && ! $force ]] 
then
  echo "Chestxray14_256 dataset already exists, skipping download. Use -f to download anyways"
else
  echo "Downloading Chestxray14_256 dataset..."
  ### CHESTXRAY14_256_DATASET ###
  wget --no-check-certificate $URL_CHESTXRAY14_256 -O "tmp.tar.gz"
  mkdir -p tmp
  tar -zxf "tmp.tar.gz" --directory tmp
  
  # move to data directory
  echo "Moving to data directory"
  rm -r -f $CHESTXRAY14_256_DATASET_DIRECTORY
  mkdir -p $CHESTXRAY14_256_DATASET_DIRECTORY
  mv tmp/*  $CHESTXRAY14_256_DATASET_DIRECTORY

  # clean up
  echo "Cleaning up"
  rm "tmp.tar.gz"
  rm -rf tmp/
fi

if [[ $all || $server ]] 
then
  if [[ -d "$CHEXPERT_FULL_DATASET_DIRECTORY" && ! $force ]] 
  then
    echo "Chexpert full dataset already exists, skipping download. Use -f to download anyways"
  else
    ### CHEXPERT_FULL_DATASET ###
    echo "Downloading Chexpert full dataset..."
    # download and unzip dev dataset
    wget --no-check-certificate $URL_CHEXPERT_FULL -O "tmp.tar.gz"
    mkdir -p tmp
    tar -zxf "tmp.tar.gz" --directory tmp
  
    # move to data directory
    echo "Moving to data directory"
    rm -r -f $CHEXPERT_FULL_DATASET_DIRECTORY
    mkdir -p $CHEXPERT_FULL_DATASET_DIRECTORY
    mv tmp/*  $CHEXPERT_FULL_DATASET_DIRECTORY

    # clean up
    echo "Cleaning up"
    rm "tmp.tar.gz"
    rm -rf tmp/
  fi
fi


if [[ $all ]] 
then
  if [[ -d "$CHESTXRAY14_512_DATASET_DIRECTORY" && ! $force ]] 
  then
    echo "Chestxray14_512 dataset already exists, skipping download. Use -f to download anyways"
  else
    ### CHESTXRAY14_512_DATASET ###
    echo "Downloading Chestxray14_512 dataset..."
    wget --no-check-certificate $URL_CHESTXRAY14_512 -O "tmp.tar.gz"
    mkdir -p tmp
    tar -zxf "tmp.tar.gz" --directory tmp
  
    # move to data directory
    echo "Moving to data directory"
    rm -r -f $CHESTXRAY14_512_DATASET_DIRECTORY
    mkdir -p $CHESTXRAY14_512_DATASET_DIRECTORY
    mv tmp/*  $CHESTXRAY14_512_DATASET_DIRECTORY

    # clean up
    echo "Cleaning up"
    rm "tmp.tar.gz"
    rm -rf tmp/
  fi
fi

echo "Finished downloading all specified datasets"