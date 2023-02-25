#!/bin/bash
# Script to save a training run to tensorboard.dev
# Args
#  --logdir: path to the log directory
#  --name: name of the run to be shown in the experiment list. Optional
#    Defaults to logdir.
#  --description: text description of the run. Optional

VALID_ARGS=$(getopt -o l:n:d --long logdir:,name:,description: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

base_dir="/media/14tb/ml/models/zetaqubit/dl/"

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -l | --logdir)
        logdir="$2"
        shift 2
        ;;
    -n | --name)
        name="$2"
        shift 2
        ;;
    -d | --description)
        description="$2"
        shift 2
        ;;
    --) shift;
        break
        ;;
  esac
done

default_name="${logdir#$base_dir} $description"
name="${name:-$default_name}"

tensorboard dev upload --logdir "$logdir" \
    --name "$name" \
    --description "$description" \
