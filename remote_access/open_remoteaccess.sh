#!/usr/bin/env bash


PASS_HASH="sha1:bbea9cdc2680:5b7c455593d9ed710a81c0fa4e5af3e91e941237"
HOST_IP=0.0.0.0
ROOT_DIR=/srv/idp-radio-1
LOG_DIR=$ROOT_DIR/nohup/

echo $LOG_DIR

mkdir -p $LOG_DIR
$ROOT_DIR/remote_access/ngrok start -config $ROOT_DIR/remote_access/tunnel.cfg --all > /dev/null &
nohup jupyter lab --ip=$HOST_IP --port=8888 --NotebookApp.allow_password_change=False --NotebookApp.password=$PASS_HASH --ContentsManager.allow_hidden=True --allow-root --no-browser --NotebookApp.iopub_data_rate_limit=10000000 > $LOG_DIR/nohup_jupyterlab.out &
#nohup jupyter notebook --ip=$HOST_IP --port=8080 --NotebookApp.allow_password_change=False  --NotebookApp.password=$PASS_HASH --allow-root --no-browser --NotebookApp.iopub_data_rate_limit=10000000 > $LOG_DIR/nohup_jupyternotebook.out &
nohup tensorboard --logdir $ROOT_DIR/models/ --host $HOST_IP --port 6006 > $LOG_DIR/nohup_tensorboard.out &
sleep 5s

while [ True ]
do 
	echo  Retrieving open ngrok tunnels...
	tunnels=$($ROOT_DIR/remote_access/get_tunnels.sh)
	echo $tunnels 
	echo
	sleep 5m
done
