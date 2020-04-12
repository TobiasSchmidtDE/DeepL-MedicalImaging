FROM python:3.7-stretch
LABEL version="1.0"

# update system
RUN apt-get update && apt-get upgrade

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /srv/idp-radio-1

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY src/ src/
RUN ls -la src/*

# Download the dataset
COPY download_dataset.sh .
RUN bash download_dataset.sh
