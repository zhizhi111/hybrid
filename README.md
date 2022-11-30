# HybridPrompt: A Topic, Sample and Attribute-aware Hybrid Prompt Framework for Visual Question Answering

Some code in this repo are copied/modified from opensource implementations made available by
[PyTorch](https://github.com/pytorch/pytorch),
[HuggingFace](https://github.com/huggingface/transformers),
[OpenNMT](https://github.com/OpenNMT/OpenNMT-py),
and [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch).
The image features are extracted using [BUTD](https://github.com/peteanderson80/bottom-up-attention).


## Requirements
We provide Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.

## Quick Start
*NOTE*: Please run `bash scripts/download_pretrained.sh $PATH_TO_STORAGE` to get our latest pretrained
checkpoints. This will download both the base and large models.

## Downstream Tasks Finetuning

### VQA
NOTE: train and inference should be ran inside the docker container
1. download data
    ```
    bash scripts/download_vqa.sh $PATH_TO_STORAGE
    ```
2. train
    ```
    horovodrun -np 4 python train_vqa.py --config config/train-vqa-base-4gpu.json \
        --output_dir $VQA_EXP
    ```
3. inference
    ```
    python inf_vqa.py --txt_db /txt/vqa_test.db --img_db /img/coco_test2015 \
        --output_dir $VQA_EXP --checkpoint 6000 --pin_mem --fp16
    ```
    The result file will be written at `$VQA_EXP/results_test/results_6000_all.json`, which can be
    submitted to the evaluation server

## Pre-tranining
download
```
bash scripts/download_indomain.sh $PATH_TO_STORAGE
```
pre-train
```
horovodrun -np 8 python pretrain.py --config config/pretrain-indomain-base-8gpu.json \
    --output_dir $PRETRAIN_EXP
```

