FROM tensorflow/tensorflow:latest-gpu-py3
LABEL version="1.0"

# update system
RUN apt-get update && apt-get upgrade -y
RUN apt-get install wget

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for container
WORKDIR  /srv/idp-radio-1

# Installing python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download the dataset. Skipped for now, as we download the directory onto the server before starting the container.
# COPY download_dataset.sh .
# RUN bash download_dataset.sh

CMD ["/bin/bash"]