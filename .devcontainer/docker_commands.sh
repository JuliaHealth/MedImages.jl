#!/bin/bash

# Build command
# Run this from the .devcontainer directory
docker build -t medimages-env .

# Run command
# This mounts the current repository to /workspace
docker run -it --rm \
    --init \
    --gpus all \
    --ipc host \
    --net host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /media/jm/hddData/projects_new/MedImages.jl:/workspace \
    medimages-env
