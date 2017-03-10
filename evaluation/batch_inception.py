#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import yaml

sys.path.insert(0, '.')
from infer import load_model  # isort:skip

parser = argparse.ArgumentParser()
parser.add_argument('--target_iter', type=int, default=100000)
parser.add_argument('--calc_iter', type=int, default=10000)
parser.add_argument('--dir_list', type=str, default='evaluation/batch_dirs.yml')
args = parser.parse_args()

prev_text = None

if len(glob.glob('results/inception_*.txt')) == 0:
    out_fn = 'results/inception_0.txt'
else:
    out_fns = []
    for fn in glob.glob('results/inception_*.txt'):
        n = int(re.search('inception_([0-9]+).txt', fn).groups()[0])
        out_fns.append((n, fn))
    out_fns = sorted(out_fns)
    out_fn = 'results/inception_{}.txt'.format(out_fns[-1][0] + 1)
    prev_fn = 'results/inception_{}.txt'.format(out_fns[-1][0])
    prev_text = [l for l in open(prev_fn).readlines()[1:] if ',' in l]
print('out_fn:', out_fn)

out_fp = open(out_fn, 'w')

# create exisiting result list
prev_results = {}
if prev_text:
    for l in prev_text:
        print(l, file=out_fp)
    for i, line in enumerate(prev_text):
        if ',' not in line:
            break
        line = [w.strip() for w in line.split(',') if w]
        prev_results[line[-1]] = line[4]

title = [
    'Inception Score (Mean)', 'Inception Score (Std)', 'Inception Score (Best)',
    'Dataset', 'Model', 'Iteration',
    'z_slow dim', 'z_fast dim',
    # 'fsgen init', 'fsgen bn use_beta',
    # 'vgen init', 'vgen bn use_beta',
    # 'vdis init', 'vdis bn use_beta',
    'Updater', 'n_frames', 'Clip', 'Parallel', 'Batchsize',
    'Result dir',
]

print(','.join(title), file=out_fp)

for dname in yaml.load(open(args.dir_list)):
    npz_fns = dict([(int(re.search('iter_([0-9]+)', fn).groups()[0]), fn)
                    for fn in glob.glob('{}/snapshot_*.npz'.format(dname))])
    if args.target_iter in npz_fns:
        target_fn = npz_fns[args.target_iter]
        target_iter = args.target_iter
    else:
        target = sorted(npz_fns.items(), key=lambda x: x[0])[-1]
        target_iter, target_fn = target
    print(target_iter, dname)

    if dname in prev_results and prev_results[dname] == str(target_iter):
        continue

    scores = []
    pool = Pool()
    for i in range(4):
        scores.append(pool.apply_async(
            subprocess.check_output,
            ([
                'python', 'evaluation/calc_inception.py',
                '--gpu', str(i), '--result_dir', dname,
                '--iter', str(target_iter), '--interpolation', 'INTER_CUBIC',
                '--calc_iter', str(args.calc_iter),
                '--seed', str(i)
            ],)))
    for i, s in enumerate(scores):
        ret = s.get()
        score = float(re.search('score:([0-9\.]+)', str(ret)).groups()[0].strip())
        scores[i] = score
        print(i, score)
    mean_score = np.mean(scores)
    best_score = np.max(scores)
    score_std = np.std(scores)

    config = yaml.load(open(glob.glob('{}/*.yml'.format(dname))[0]))
    config['models']['video_generator']['conv_ch'] = 512
    config['models']['video_generator']['wscale'] = 0.01
    fsgen = load_model(dname, config, 'frame_seed_generator')
    vgen = load_model(dname, config, 'video_generator')
    vdis = load_model(dname, config, 'video_discriminator')

    updater_name = config['updater']['name']
    if updater_name == 'TGANUpdaterWGAN':
        model, clip, parallel = 'WGAN', 'SVC', 'FALSE'
    elif updater_name == 'TGANUpdaterWGANClip':
        model, clip, parallel = 'WGAN', 'Clip', 'FALSE'
    elif updater_name == 'TGANUpdaterWGANSVC':
        model, clip, parallel = 'WGAN', 'SVC', 'FALSE'
    elif updater_name == 'TGANUpdaterWGANParallelSVC':
        model, clip, parallel = 'WGAN', 'SVC', 'TRUE'
    elif updater_name == 'TGANUpdaterWGANVideoFrameDis':
        model, clip, parallel = 'WGAN', 'SVC', 'TRUE'
    elif updater_name == 'TGANUpdaterVanilla':
        model, clip, parallel = 'Vanilla', 'None', 'FALSE'
    else:
        raise ValueError(updater_name)

    info = {
        'Dataset': config['dataset']['dataset_name'],
        'Updater': updater_name,
        'Model': model,
        'Iteration': str(target_iter),
        'Clip': clip,
        'Parallel': parallel,
        'Batchsize': str(config['batchsize']),
        'z_slow dim': str(config['models']['frame_seed_generator']['args']['z_slow_dim']),
        'z_fast dim': str(config['models']['frame_seed_generator']['args']['z_fast_dim']),
        'n_frames': str(config['dataset']['args']['n_frames']),
        # 'fsgen init': fsgen.dc0.initializer_name if hasattr(fsgen.dc0, 'initializer_name') else 'None',
        # 'fsgen bn use_beta': 'TRUE' if fsgen.bn0.use_beta else 'FALSE',
        # 'vgen init': vgen.dc1.initializer_name,
        # 'vgen bn use_beta': 'TRUE' if vgen.bn1.use_beta else 'FALSE',
        # 'vdis init': vdis.c0.initializer_name,
        # 'vdis bn use_beta': 'TRUE' if vdis.bn0.use_beta else 'FALSE',
        'Inception Score (Mean)': '{}'.format(mean_score),
        'Inception Score (Std)': '{}'.format(score_std),
        'Inception Score (Best)': '{}'.format(best_score),
        'Result dir': dname,
    }

    print(','.join([info[key] for key in title]), file=out_fp)
    print('mean:', mean_score, '+-', score_std, 'best:', best_score)
out_fp.close()
