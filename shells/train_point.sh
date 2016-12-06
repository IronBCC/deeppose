#!/bin/bash
# Copyright (c) 2016 Shunta Saito

CHAINER_TYPE_CHECK=0 \
python3 scripts/train.py \
--model models/ResNet50.py \
--gpus -1 \
--epoch 10 \
--batchsize 64 \
--snapshot 4 \
--valid_freq 10 \
--train_csv_fn data/Point/points.csv \
--test_csv_fn data/Point/points.csv \
--img_dir data/Point/imgs \
--test_freq 10 \
--seed 1701 \
--im_size 220 \
--rotate \
--rotate_range 20 \
--translate \
--translate_range 10 \
--n_joints 32 \
--fname_index 0 \
--joint_index 1 \
--opt Adam
