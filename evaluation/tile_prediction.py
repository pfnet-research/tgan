#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

import chainer
import cv2 as cv
from c3d_ft import C3DVersion1
from chainer import Variable

sys.path.insert(0, '.')
from infer import get_models  # isort:skip
from infer import vgen_forward  # isort:skip


def calc_inception(ys):
    p_all = np.mean(ys, axis=0, keepdims=True)
    kl = ys * np.log(ys + 1e-7) - ys * np.log(p_all + 1e-7)
    return np.exp(kl)


def make_video(fsgen, vgen, n=48):
    xp = fsgen.xp

    with chainer.using_config('train', False):
        z_slow = xp.random.uniform(-1, 1, (n, fsgen.z_slow_dim)).astype('f')
        z_slow = Variable(z_slow)
        z_fast = fsgen(z_slow)
        y = vgen_forward(vgen, z_slow, z_fast)
        y = y.reshape(n, -1, y.shape[1], y.shape[2], y.shape[3])
        y = y.transpose(0, 2, 1, 3, 4)

    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--iter', type=int, default=100000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='infer')
    parser.add_argument('--mean', type=str, default='/mnt/sakura201/mitmul/codes/tgan2_orig/inception/mean2.npz')
    parser.add_argument('--interpolation', type=str, default='INTER_CUBIC')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=48)
    args = parser.parse_args()

    np.random.seed(args.seed)
    chainer.cuda.cupy.random.seed(args.seed)
    inter_method = args.interpolation
    args.interpolation = getattr(cv, args.interpolation)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    chainer.cuda.Device(args.gpu).use()
    xp = chainer.cuda.cupy

    fsgen, vgen, _ = get_models(args.result_dir, args.iter)
    if args.gpu >= 0:
        fsgen.to_gpu()
        vgen.to_gpu()

    c3dmodel = C3DVersion1()
    c3dmodel.to_gpu()
    mean = np.load(args.mean)['mean'].astype('f')
    mean = mean.reshape((3, 1, 16, 128, 171))[:, :, :, :, 21:21 + 128]

    videos = []
    for i in range(100):
        gen_video = make_video(fsgen, vgen, args.batchsize)
        n, c, f, h, w = gen_video.shape
        x = gen_video.transpose(0, 2, 3, 4, 1).reshape(n * f, h, w, c)
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

        score = calc_inception(y)

        score = score.max(axis=1)
        video = gen_video[score.argmax()]
        video = video.transpose(1, 2, 3, 0) * 128 + 128
        videos.append(video)
        print(i)

    videos = np.array(videos)
    n, f, h, w, c = videos.shape
    y = videos.transpose(1, 0, 2, 3, 4)
    hn = int(np.sqrt(n))
    y = y.reshape(f, hn, hn, h, w, c)
    y = y.transpose(0, 1, 3, 2, 4, 5)
    y = y.reshape(f, hn * h, hn * w, c)
    print(y.shape)

    prefix = os.path.basename(args.result_dir)
    for i, p in enumerate(y):
        fn = '{}/{}_seed-{}_{}.png'.format(args.out_dir, prefix, args.seed, i)
        cv.imwrite(fn, p[:, :, ::-1])
    fn = '{}/{}_seed-{}.avi'.format(args.out_dir, prefix, args.seed)
    subprocess.call([
        'ffmpeg', '-i', '{}/{}_seed-{}_%d.png'.format(args.out_dir, prefix, args.seed),
        '-vcodec', 'rawvideo', '-pix_fmt', 'yuv420p', fn])
    for _fn in glob.glob('{}/{}_*.png'.format(args.out_dir, prefix)):
        os.remove(_fn)
