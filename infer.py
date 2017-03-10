#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import imp
import os
import subprocess
import sys

import numpy as np
import yaml

import chainer
import chainer.functions as F
import cv2 as cv
from chainer import Variable
from chainer import serializers


def make_video(fsgen, vgen, n, z_slow=None, z_fast=None):
    xp = fsgen.xp

    if z_slow is None:
        z_slow = xp.random.uniform(-1, 1, (n, fsgen.z_slow_dim)).astype('f')
    z_slow = Variable(z_slow)

    with chainer.using_config('train', False):
        if z_fast is None:
            z_fast = fsgen(z_slow)
        y = vgen_forward(vgen, z_slow, z_fast)
        y = y.reshape(n, -1, y.shape[1], y.shape[2], y.shape[3])
        y = y.transpose(0, 2, 1, 3, 4)

        z_slow = chainer.cuda.to_cpu(z_slow.data)
        z_fast = chainer.cuda.to_cpu(z_fast.data)

    return y, z_slow, z_fast


def vgen_forward(vgen, z_slow, z_fast):
    B, n_z_fast, n_frames = z_fast.shape
    z_fast = F.reshape(F.transpose(
        z_fast, [0, 2, 1]), (B * n_frames, n_z_fast))

    B, n_z_slow = z_slow.shape
    z_slow = F.reshape(F.broadcast_to(F.reshape(
        z_slow, (B, 1, n_z_slow)), (B, n_frames, n_z_slow)),
        (B * n_frames, n_z_slow))

    with chainer.using_config('train', False):
        img_fake = vgen(z_slow, z_fast)
    return chainer.cuda.to_cpu(img_fake.data)


def load_model(result_dir, config, model_type, snapshot_path=None):
    model_fn = '{}/{}'.format(result_dir, os.path.basename(config['models'][model_type]['fn']))
    model_name = config['models'][model_type]['name']
    kwargs = config['models'][model_type]['args']
    model = imp.load_source(model_name, model_fn)
    model = getattr(model, model_name)(**kwargs)
    if snapshot_path:
        serializers.load_npz(snapshot_path, model)
    return model


def get_models(result_dir, n_iter):
    config = yaml.load(open(glob.glob('{}/*.yml'.format(result_dir))[0]))
    fsgen = load_model(result_dir, config, 'frame_seed_generator', '{}/gen_iter_{}.npz'.format(result_dir, n_iter))
    vgen = load_model(result_dir, config, 'video_generator', '{}/vgen_iter_{}.npz'.format(result_dir, n_iter))
    vdis = load_model(result_dir, config, 'video_discriminator', '{}/vdis_iter_{}.npz'.format(result_dir, n_iter))
    return fsgen, vgen, vdis


def save_video(y, seed, out_dir='infer', prefix=''):
    y = y.transpose(0, 2, 3, 4, 1)
    n, f, h, w, c = y.shape
    y = y.transpose(1, 0, 2, 3, 4)
    hn = int(np.sqrt(n))
    y = y.reshape(f, hn, hn, h, w, c)
    y = y.transpose(0, 1, 3, 2, 4, 5)
    y = y.reshape(f, hn * h, hn * w, c)
    for i, p in enumerate(y):
        fn = '{}/{}_seed-{}_{}.png'.format(out_dir, prefix, seed, i)
        cv.imwrite(fn, p[:, :, ::-1])
    fn = '{}/{}_seed-{}.avi'.format(out_dir, prefix, seed)
    subprocess.call([
        'ffmpeg', '-i', '{}/{}_seed-{}_%d.png'.format(out_dir, prefix, seed),
        '-vcodec', 'rawvideo', '-pix_fmt', 'yuv420p', fn])
    # subprocess.call([
    #     'ffmpeg', '-i', '{}.avi'.format(os.path.splitext(fn)[0]),
    #     '-vcodec', 'libx264', fn.replace('.avi', '.mp4')])
    for _fn in glob.glob('{}/{}_*.png'.format(out_dir, prefix)):
        os.remove(_fn)
    # os.remove('{}.avi'.format(os.path.splitext(fn)[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--iter', type=int, default=100000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='infer')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--video', action='store_true', default=False)
    parser.add_argument('--images', action='store_true', default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    chainer.cuda.cupy.random.seed(args.seed)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    chainer.cuda.Device(args.gpu).use()

    fsgen, vgen, _ = get_models(args.result_dir, args.iter)
    if args.gpu >= 0:
        fsgen.to_gpu()
        vgen.to_gpu()

    y, z_slow, z_fast = make_video(fsgen, vgen, n=args.n)
    y = y * 128 + 128

    if args.video:
        save_video(y, args.seed, args.out_dir, prefix=os.path.basename(args.result_dir))

    if args.images:
        # sf = [0, 3, 6, 9, 12, 15]
        sf = list(range(16))
        base_dname = os.path.basename(args.result_dir)
        n, c, f, h, w = y.shape
        videos = y.transpose(0, 2, 3, 4, 1)  # n, f, h, w, c
        for i, video in enumerate(videos):
            video = video[sf, ...]  # f, h, w, c
            video = video.transpose(1, 0, 2, 3)  # h, f, w, c
            video = video.reshape(h, len(sf) * w, c)
            fn = '{}/{}_seed-{}_{}.png'.format(args.out_dir, base_dname, args.seed, i)
            cv.imwrite(fn, video[:, :, ::-1])


if __name__ == '__main__':
    sys.exit(main())
