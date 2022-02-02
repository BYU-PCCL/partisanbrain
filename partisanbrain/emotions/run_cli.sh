#!/bin/bash

xhost +local:root

sudo docker run -it \
  --rm \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/wingated:/home/wingated \
  -w `pwd` \
  --net host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel /bin/bash

#  --net host\
#  --device=/dev/video1:/dev/video0 \

#  -p 11491:11491/udp \
#  -p 11490:11490/udp \
#  -p 5678:5678/udp \
