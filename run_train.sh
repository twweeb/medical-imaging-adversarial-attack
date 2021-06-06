#!/bin/bash

python3 train.py --arch resnet18 \
        --dataset_dir dataset/MedNIST \
        --batch_size 256 --epoch 20 --model_dir pretrained_model/