#!/usr/bin/env bash

mkdir -p /srv/idp-radio-1/nohup/
/srv/idp-radio-1/remote_access/ngrok start -config tunnel.cfg --all > /dev/null &
nohup jupyter lab --ip=0.0.0.0 --port=8888 --NotebookApp.allow_password_change=False --NotebookApp.password="sha1:bbea9cdc2680:5b7c455593d9ed710a81c0fa4e5af3e91e941237" --ContentsManager.allow_hidden=True --allow-root --no-browser > /srv/idp-radio-1/nohup/nohup_jupyterlab.out &
nohup jupyter notebook --ip=0.0.0.0 --port=8080 --NotebookApp.allow_password_change=False  --NotebookApp.password="sha1:bbea9cdc2680:5b7c455593d9ed710a81c0fa4e5af3e91e941237" --allow-root --no-browser > /srv/idp-radio-1/nohup/nohup_jupyternotebook.out &
nohup tensorboard --logdir /srv/idp-radio-1/models/ --host 0.0.0.0 --port 6006 > /srv/idp-radio-1/nohup/nohup_tensorboard.out &
echo
/srv/idp-radio-1/remote_access/get_tunnels.sh
