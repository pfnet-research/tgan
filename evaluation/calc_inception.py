#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np

import chainer
import chainer.cuda
import cv2 as cv
from c3d_ft import C3DVersion1
from chainer import Variable
from chainer import cuda
from tqdm import tqdm

sys.path.insert(0, '.')  # isort:skip
from infer import get_models  # isort:skip
from infer import make_video  # isort:skip


def calc_inception(ys):
    N, C = ys.shape
    p_all = np.mean(ys, axis=0, keepdims=True)
    kl = np.sum(ys * np.log(ys + 1e-7) - ys * np.log(p_all + 1e-7)) / N
    return np.exp(kl)


def main():
    parser = argparse.ArgumentParser(description='inception score')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--iter', type=int, default=100000)
    parser.add_argument('--calc_iter', type=int, default=10000)
    # parser.add_argument('--mean', type=str, default='/mnt/sakura201/mitmul/codes/tgan2_orig/inception/mean2.npz')
    parser.add_argument('--mean', type=str, default='mean2.npz')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--interpolation', type=str, default='INTER_CUBIC')
    args = parser.parse_args()

    np.random.seed(args.seed)

    inter_method = args.interpolation
    args.interpolation = getattr(cv, args.interpolation)

    cuda.get_device(args.gpu).use()
    chainer.cuda.cupy.random.seed(args.seed)
    xp = chainer.cuda.cupy

    c3dmodel = C3DVersion1()
    c3dmodel.to_gpu()

    # load model
    fsgen, vgen, _ = get_models(args.result_dir, args.iter)
    if args.gpu >= 0:
        fsgen.to_gpu()
        vgen.to_gpu()
    batchsize = 48

    mean = np.load(args.mean)['mean'].astype('f')
    mean = mean.reshape((3, 1, 16, 128, 171))[:, :, :, :, 21:21 + 128]

    # generator
    ys = []
    for i in tqdm(range(args.calc_iter // batchsize)):
        x, _, _ = make_video(fsgen, vgen, batchsize)
        n, c, f, h, w = x.shape
        x = x.transpose(0, 2, 3, 4, 1).reshape(n * f, h, w, c)
        x = x * 128 + 128
        x_ = np.zeros((n * f, 128, 128, 3))
        for t in range(n * f):
            x_[t] = np.asarray(
                cv.resize(x[t], (128, 128), interpolation=args.interpolation))
        x = x_.transpose(3, 0, 1, 2).reshape(3, n, f, 128, 128)
        x = x[::-1] - mean  # mean file is BGR-order while model outputs RGB-order
        x = x[:, :, :, 8:8 + 112, 8:8 + 112].astype('f')
        x = x.transpose(1, 0, 2, 3, 4)
        with chainer.using_config('train', False) and \
                chainer.no_backprop_mode():
            # C3D takes an image with BGR order
            y = c3dmodel(Variable(xp.asarray(x)),
                         layers=['prob'])['prob'].data.get()
            ys.append(y)
    ys = np.asarray(ys).reshape((-1, 101))

    score = calc_inception(ys)
    with open('{}/inception_iter-{}_{}.txt'.format(args.result_dir, args.iter, inter_method), 'w') as fp:
        print(args.result_dir, args.iter, args.calc_iter, args.mean, score, file=fp)
        print(args.result_dir, args.iter, 'score:{}'.format(score))

    return 0


if __name__ == '__main__':
    sys.exit(main())
