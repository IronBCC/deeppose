import random

import chainer

import cmd_options
import dataset
import imp
import logger
import logging
import loss
import os
import shutil
import sys
import tempfile
import time

import argparse
import chainer
import cv2 as cv
import numpy as np
import time

import loss
from results.AlexNet import AlexNet
from chainer import serializers
from chainer.cuda import to_gpu

def get_model(gpu, n_joins):
    print("Model loading...")
    model = AlexNet(n_joins)
    model = loss.PoseEstimationError(model)

    model.predictor.train = False
    model.train = False
    serializers.load_npz('results/epoch-220.model', model)
    model = model.predictor
    print("Model loaded.")
    return model


def draw_joints(image, joints, ignore_joints):
    if image.shape[2] != 3:
        _image = image.transpose(1, 2, 0).astype(np.uint8).copy()
    else:
        _image = image.copy()
    for point in joints:
        cv.circle(_image, (point[0], point[1]), 6, (255, 0, 0), 3)

    return _image


def convert_joins(joints):
    if(len(joints.shape) == 2 & joints.shape[0] == 2):
        return joints
    if(joints.shape[0] == 1):
        joints = joints[0]

    joints = np.array(list(zip(joints.data[0::2], joints.data[1::2])))
    joints = joints.astype(np.int32)
    return joints

def transformImage(image, resize=220):
    # return image
    return image.transpose((2, 0, 1)).astype(np.float32)

if __name__ == '__main__':
    args = cmd_options.get_arguments()
    model = get_model(args.gpu, args.n_joints)

    train_dataset = dataset.PoseDataset(
        args.train_csv_fn, args.img_dir, args.im_size, args.fliplr,
        args.rotate, args.rotate_range, args.zoom, args.base_zoom,
        args.zoom_range, args.translate, args.translate_range, args.min_dim,
        args.coord_normalize, args.gcn, args.n_joints, args.fname_index,
        args.joint_index, args.symmetric_joints, args.ignore_label,
        1000
    )

    while (True):
        origin, bbox_pred, ignored = train_dataset.get_example( random.randint(0, len(train_dataset) - 1))
        img = origin
        img = np.expand_dims(img, axis=0)
        joints = model(img)

        joints = convert_joins(joints)
        result = draw_joints(origin, joints, ignored)
        cv.imshow('frame', result)
        time.sleep(2)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()