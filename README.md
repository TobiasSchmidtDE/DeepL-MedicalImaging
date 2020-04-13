# IDP Radio

## Getting Started

1. Clone this repository

2. Use the `download_dataset.sh` script to download the dev-dataset.

## Deploy using Docker

1. Clone this repository

2. Install [Docker](https://docs.docker.com/engine/install/ubuntu/)

3. Build the Docker Image: `docker build --tag idp-radio .` This will build a docker image based on the tensorflow image, install all dependecies and download the dataset into the container. 

4. Start and run commands in the Docker container: `docker run -v $PWD/src:/srv/idp-radio-1/src --gpus all --user $(id -u):$(id -g) -it idp-radio python3 src/main.py`.
  - `-v` mounts the src folder into the docker container
  - `--gpus all` enables all GPUs
  - The working directory inside the container is ` /srv/idp-radio-1` and the dataset is available in `/srv/idp-radio-1/data`.

5. Run commands in the container if its already running `docker exec -it container_name command`
