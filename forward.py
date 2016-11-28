#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import serializers
from chainer.cuda import to_gpu
# from lib.cpu_nms import cpu_nms as nms
# from lib.models.faster_rcnn import FasterRCNN

import argparse
import chainer
import cv2 as cv
import numpy as np
import time

import loss
from results.AlexNet import AlexNet
from transform import Transform

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


def get_model(gpu, n_joins):
    print("Model loading...")
    model = AlexNet(n_joins)
    model = loss.PoseEstimationError(model)

    model.predictor.train = False
    model.train = False
    serializers.load_npz('results/epoch-50.model', model)
    model = model.predictor
    print("Model loaded.")
    return model


def img_preprocessing(orig_img, pixel_means, max_size=220, scale=220):
    img = orig_img.astype(np.float32, copy=True)
    # img -= pixel_means
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    im_scale = float(scale) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv.INTER_LINEAR)

    return img.transpose([2, 0, 1]).astype(np.float32), im_scale


def revert(img, joints):
    h, w, c = img.shape
    center_pt = np.array([w / 2, h / 2])

    joints = np.array(list(zip(joints.data[0::2], joints.data[1::2])))  # x,y order
    # joints[:, 0] *= w / 2
    # joints[:, 1] *= h / 2
    # joints *= w / 2
    # joints += h / 2
    # joints += center_pt
    joints = joints.astype(np.int32)

    # if self.args.gcn:
    #     img -= img.min()
    #     img /= img.max()
    #     img *= 255
    #     img = img.astype(np.uint8)

    return img, joints

def draw_joints(image, joints, ignore_joints):
    if image.shape[2] != 3:
        _image = image.transpose(1, 2, 0).copy()
    else:
        _image = image.copy()
    # _image = image.copy()

    # if joints.ndim == 1:
    #     joints = np.array(list(zip(joints[0::2], joints[1::2])))
    # if ignore_joints.ndim == 1:
    #     ignore_joints = np.array(
    #         list(zip(ignore_joints[0::2], ignore_joints[1::2])))
    for point in joints:
        # if ignore_joints is not None \
        #         and (ignore_joints[i][0] == 0 or ignore_joints[i][1] == 0):
        #     continue
        cv.circle(_image, (point[0], point[1]), 6, (255, 0, 0), 3)
        # cv.putText(
        #     _image, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX,
        #     1.0, (255, 255, 255), 3)
        # cv.putText(
        #     _image, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX,
        #     1.0, (0, 0, 0), 1)
    return _image
    # _, fn_img = tempfile.mkstemp()
    # basename = os.path.basename(fn_img)
    # fn_img = fn_img.replace(basename, prefix + basename)
    # fn_img = fn_img + '.png'
    # cv.imwrite(fn_img, _image)


def draw_result(out, im_scale, clss, bbox, nms_thresh, conf):
    CV_AA = 16
    for cls_id in range(1, 21):
        _cls = clss[:, cls_id][:, np.newaxis]
        _bbx = bbox[:, cls_id * 4: (cls_id + 1) * 4]
        dets = np.hstack((_bbx, _cls))
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= conf)[0]
        for i in inds:
            x1, y1, x2, y2 = map(int, dets[i, :4])
            cv.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2, CV_AA)
            ret, baseline = cv.getTextSize(
                CLASSES[cls_id], cv.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            cv.rectangle(out, (x1, y2 - ret[1] - baseline),
                         (x1 + ret[0], y2), (0, 0, 255), -1)
            cv.putText(out, CLASSES[cls_id], (x1, y2 - baseline),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, CV_AA)

    return out


def transformImage(image, resize=220):
    if (resize > 0):
        image = cv.resize(image, (resize, resize),
                          interpolation=cv.INTER_NEAREST)
    # return image
    return image.transpose((2, 0, 1)) #.astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nms_thresh', type=float, default=0.3)
    parser.add_argument('--conf', type=float, default=0.8)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--n_joints', type=int, default=7)
    args = parser.parse_args()

    xp = chainer.cuda.cupy if chainer.cuda.available and args.gpu >= 0 else np
    model = get_model(args.gpu, args.n_joints)
    if chainer.cuda.available and args.gpu >= 0:
        model.to_gpu(args.gpu)


    cap = cv.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        # ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # orig_image = cv.imread("results/test_img.jpg")
        ret, orig_image = cap.read();
        start = time.clock()
        # img, im_scale = img_preprocessing(orig_image, PIXEL_MEANS)
        img = transformImage(orig_image)

        img = draw_joints(img, [], None)
        cv.imshow('frame', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        continue

        transImg = img
        print("img_preprocessing = ", (time.clock() - start))

        start = time.clock()
        img = np.expand_dims(img, axis=0)
        if args.gpu >= 0:
            img = to_gpu(img, device=args.gpu)

        img = chainer.Variable(img, volatile=True)
        # h, w = img.data.shape[2:]
        print("variable = ", (time.clock() - start))

        start = time.clock()
        bbox_pred = model(img)
        bbox_pred = bbox_pred[0]
        print("recognition = ", (time.clock() - start))

        # if args.gpu >= 0:
            # cls_score = chainer.cuda.cupy.asnumpy(cls_score)
            # bbox_pred = chainer.cuda.cupy.cupyasnumpy(bbox_pred)

        img, bbox_pred = revert(transImg, bbox_pred)
        result = draw_joints(img, bbox_pred, None)
        # result = draw_result(orig_image, im_scale, cls_score, bbox_pred,
        #                      args.nms_thresh, args.conf)
        # print('%d (%d) found' % (len(found_filtered), len(found)))
        cv.imshow('frame', result)

        # Display the resulting frame
        # cv2.imshow('frame', gray)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
