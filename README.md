# IDP Radio

## Getting Started

1. Clone this repository

2. Use the `download_dataset.sh` script to download the dev-dataset.

## Deploy using Docker

1. Clone this repository

2. Insall [https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/)

3. Build the Docker image using `docker build --tag idp-radio .`

4. Start the Docker container `docker run  -d -t --name radio1 idp-radio:latest`

5. Run commands in the container `docker exec -it radio1 python3 src/main.py`
