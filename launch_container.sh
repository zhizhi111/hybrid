# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TXT_DB="/home/yzh/vqa"
IMG_DIR="/home/hky/processed_data_and_pretrained_models/img_db"
OUTPUT="/data/yzh/workspace/vqa_output"
#PRETRAIN_DIR="/home/hky/processed_data_and_pretrained_models/pretrained"
PRETRAIN_DIR="/home/yzh/pretrained_model/base_for_vqa"
ANN_DIR="/home/hky/processed_data_and_pretrained_models/ann"

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES=3
fi


docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --name yzh3 --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/txt,type=bind \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src hky/uniter:v1 
