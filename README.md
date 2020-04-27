# IDP Radio

## Getting Started

1. Clone this repository

2. Use the `download_dataset.sh` script to download the dev dataset. Use `-f` to overwrite the existing dataset and/or `-a` to download both the dev- and full dataset.

## Deploy using Docker

1. Clone this repository

2. Install [Docker](https://docs.docker.com/engine/install/ubuntu/)

3. Build the Docker Image: `docker build --tag idp-radio .` This will build a docker image based on the tensorflow image, install all dependecies and download the dataset into the container. 

4. Start and run commands in the Docker container: `docker run -v $PWD:/srv/idp-radio-1 --gpus all -it idp-radio /bin/bash`.
  - `-v` mounts the src folder into the docker container
  - `--gpus all` enables all GPUs
  - The working directory inside the container is ` /srv/idp-radio-1` 

5. Run commands in the container if its already running `docker exec -it container_name command`
