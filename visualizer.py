import os

import subprocess
import argparse
import chainer
import chainer.cuda
import chainer.functions as F
import numpy as np
from chainer import Variable
from PIL import Image


def out_generated_movie(fsgen, vgen, vdis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        xp = fsgen.xp

        with chainer.using_config('train', False) and \
                chainer.no_backprop_mode():
            z_slow = xp.random.uniform(
                -1, 1, (rows, fsgen.z_slow_dim)).astype('f')
            z_slow = Variable(z_slow)

            z_fast = fsgen(z_slow)
            B, n_z_fast, n_frames = z_fast.shape
            z_fast = F.reshape(F.transpose(
                z_fast, [0, 2, 1]), (B * n_frames, n_z_fast))

            B, n_z_slow = z_slow.shape
            z_slow = F.reshape(F.broadcast_to(F.reshape(
                z_slow, (B, 1, n_z_slow)), (B, n_frames, n_z_slow)),
                (B * n_frames, n_z_slow))

            img_fake = vgen(z_slow, z_fast)

        x = chainer.cuda.to_cpu(img_fake.data)

        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C == 1:
                x = x.reshape((rows * H, cols * W))
            else:
                x = x.reshape((rows * H, cols * W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image_{}_{:0>8}.png'.format(name, trainer.updater.iteration)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x, mode=mode).convert('RGB').save(preview_path)

        x = np.asarray(x * 127 + 127, dtype=np.uint8)
        save_image(x, "img")
        _, C, H, W = x.shape
        x = x.transpose(0, 2, 3, 1)  # N, H, W, C
        save_mov(
            x, rows, cols, H, W, C,
            '{}/preview/video_{:0>8}.avi'.format(dst, trainer.updater.iteration))
    return make_image


def save_mov(img, rows, cols, H, W, C, fn):
    img = img.reshape(rows, cols, H, W, C)
    img = img.transpose(1, 0, 2, 3, 4)
    img = img.reshape(cols,
                      int(np.sqrt(rows)), int(np.sqrt(rows)),
                      H, W, C)
    img = img.transpose(0, 1, 3, 2, 4, 5)
    img = img.reshape(cols, H * int(np.sqrt(rows)),
                      W * int(np.sqrt(rows)), C)

    out_dir = os.path.splitext(fn)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for frame_i, frame in enumerate(img):
        if frame.ndim == 2 or frame.shape[2] == 1:
            frame = np.broadcast_to(frame, (frame.shape[0], frame.shape[1], 3))
        frame = frame.astype(np.uint8)
        Image.fromarray(frame).save(
            '{}/{}.png'.format(out_dir, frame_i))

    subprocess.call([
        'ffmpeg', '-r', '10', '-i',
        '{}/%d.png'.format(out_dir),
        '-qscale', '0', '-s', '640x640', fn])
    subprocess.call([
        'ffmpeg', '-i', '{}.avi'.format(os.path.splitext(fn)[0]),
        fn.replace('.avi', '.mov')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview_png', type=str)
    parser.add_argument('--H', type=int, default=64)
    parser.add_argument('--W', type=int, default=64)
    parser.add_argument('--C', type=int, default=3)
    parser.add_argument('--rows', type=int, default=100)
    parser.add_argument('--cols', type=int, default=16)
    args = parser.parse_args()

    out_dir = os.path.splitext(args.preview_png)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img = np.asarray(Image.open(args.preview_png))
    img = img.reshape(args.rows, args.H, args.cols, args.W, args.C)
    img = img.transpose(2, 0, 1, 3, 4)  # cols, rows, H, W, C
    img = img.reshape(args.cols,
                      int(np.sqrt(args.rows)), int(np.sqrt(args.rows)),
                      args.H, args.W, args.C)
    img = img.transpose(0, 1, 3, 2, 4, 5)
    img = img.reshape(args.cols, args.H * int(np.sqrt(args.rows)),
                      args.W * int(np.sqrt(args.rows)), args.C)

    for frame_i, frame in enumerate(img):
        Image.fromarray(
            frame.astype(np.uint8)).save(
            '{}/{}.png'.format(out_dir, frame_i))

    subprocess.call([
        'ffmpeg', '-r', '10', '-i',
        '{}/%d.png'.format(out_dir),
        '-qscale', '0', '-s', '640x640', '{}/out.avi'.format(out_dir)])
    subprocess.call([
        'ffmpeg', '-i', '{}/out.avi'.format(out_dir),
        '{}/out.mov'.format(out_dir)])
