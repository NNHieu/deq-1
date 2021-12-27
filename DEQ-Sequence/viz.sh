#!/bin/bash
export DOCKER_CONTAINER_ID=5ca8c2ea3df3
docker start $DOCKER_CONTAINER_ID
docker exec $DOCKER_CONTAINER_ID cd DEQ-Sequence & tools/tmp.py
tmux new-session -d -s rank0 docker exec $DOCKER_CONTAINER_ID cd DEQ-Sequence & CUDA_VISIBLE_DEVICES '0,' & bash wt103_deq_transformer.sh lossland --f_thres 30 --pretrain_step 0 --rect -1 -1 1 1 --resolution 31 31 --rank 0 --nproc 2
tmux new-session -d -s rank1 docker exec $DOCKER_CONTAINER_ID cd DEQ-Sequence & CUDA_VISIBLE_DEVICES '1,' & bash wt103_deq_transformer.sh lossland --f_thres 30 --pretrain_step 0 --rect -1 -1 1 1 --resolution 31 31 --rank 1 --nproc 2