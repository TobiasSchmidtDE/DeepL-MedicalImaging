# #!/bin/bash

URL=https://storage.googleapis.com/idp-chexpert-data/dev_dataset.zip
TMPFILE=`mktemp`

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