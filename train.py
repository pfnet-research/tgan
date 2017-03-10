#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib  # isort:skip
matplotlib.use('Agg')  # isort:skip

import argparse
import os
import shutil
import sys
import time

import chainer
import yaml
from chainer import training
from chainer.training import extensions

from visualizer import out_generated_movie


class Config(object):

    def __init__(self, config_dict):
        self.config = config_dict

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise AttributeError(key)

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)


def load_module(fn, name):
    mod_name = os.path.splitext(os.path.basename(fn))[0]
    mod_path = os.path.dirname(fn)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)


def load_dataset(config):
    dataset = load_module(config.dataset['dataset_fn'],
                          config.dataset['dataset_name'])
    return dataset(**config.dataset['args'])


def load_model(model_fn, model_name, args=None):
    model = load_module(model_fn, model_name)
    if args:
        return model(**args)
    return model()


def load_models(config):
    fsgen_conf = config.models['frame_seed_generator']
    fsgen = load_model(fsgen_conf['fn'], fsgen_conf['name'], fsgen_conf['args'])
    vgen_conf = config.models['video_generator']
    vgen = load_model(vgen_conf['fn'], vgen_conf['name'], vgen_conf['args'])
    vdis_conf = config.models['video_discriminator']
    vdis = load_model(vdis_conf['fn'], vdis_conf['name'], vdis_conf['args'])
    return fsgen, vgen, vdis


def load_updater_class(config):
    return load_module(config.updater['fn'], config.updater['name'])


def create_result_dir(config_path, config):
    if not hasattr(config, 'result_dir'):
        config_fn = os.path.splitext(os.path.basename(config_path))[0]
        config.result_dir = 'results/{}_{}_0'.format(
            config_fn, time.strftime('%Y-%m-%d_%H-%M-%S'))
        if os.path.exists(config.result_dir):
            config.result_dir[-1] = str(int(config.result_dir.split('_')[-1]) + 1)
        if not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

    copy_to_result_dir(config_path, config.result_dir)
    copy_to_result_dir(
        config.models['frame_seed_generator']['fn'], config.result_dir)
    copy_to_result_dir(
        config.models['video_generator']['fn'], config.result_dir)
    copy_to_result_dir(
        config.models['video_discriminator']['fn'], config.result_dir)
    copy_to_result_dir(
        config.dataset['dataset_fn'], config.result_dir)
    copy_to_result_dir(
        config.updater['fn'], config.result_dir)
    return config.result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    config = Config(yaml.load(open(args.config_path)))
    dataset = load_dataset(config)
    fsgen, vgen, vdis = load_models(config)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        fsgen.to_gpu()
        vgen.to_gpu()
        vdis.to_gpu()

    def make_optimizer(model, alpha=0.00005, beta1=0.5):
        optimizer = chainer.optimizers.RMSprop(lr=alpha)
        optimizer.setup(model)
        return optimizer

    opt_vgen = make_optimizer(vgen)
    opt_vdis = make_optimizer(vdis)
    opt_fsgen = make_optimizer(fsgen)

    iterator = chainer.iterators.MultiprocessIterator(dataset, config.batchsize)
    updater = load_updater_class(config)
    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'models': (fsgen, vgen, vdis),
        'iterator': iterator,
        'optimizer': {'fsgen': opt_fsgen, 'vgen': opt_vgen, 'vdis': opt_vdis},
        'device': args.gpu
    })
    updater = updater(**kwargs)
    out = create_result_dir(args.config_path, config) if not args.test else 'results/test'
    print(out)
    trainer = training.Trainer(updater, (config.epoch, 'epoch'), out=out)

    snapshot_interval = (config.snapshot_interval, 'iteration')
    display_interval = (config.display_interval, 'iteration')

    # Snapshot
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        fsgen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        vgen, 'vgen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        vdis, 'vdis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)

    # Logging
    trainer.extend(extensions.LogReport(trigger=display_interval))
    if 'vanilla' not in args.config_path:
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'vdis/loss_gen', 'vdis/loss_dis', 'elapsed_time'
        ]), trigger=display_interval)
    else:
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'vdis/loss_dis_fake', 'vdis/loss_dis_real', 'elapsed_time'
        ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=config.display_interval))
    trainer.extend(extensions.PlotReport(
        ['vdis/loss_gen'], trigger=display_interval, file_name='loss_gen.png'),
        trigger=display_interval)
    trainer.extend(extensions.PlotReport(
        ['vdis/loss_dis'], trigger=display_interval, file_name='loss_dis.png'),
        trigger=display_interval)

    # Save movie
    trainer.extend(
        out_generated_movie(fsgen, vgen, vdis, 100, 16, config.seed, out),
        trigger=snapshot_interval)

    # Resume from a snapshot
    if hasattr(config, 'resume'):
        chainer.serializers.load_npz(config.resume, trainer)

    # Run the training
    trainer.run()

    return 0

if __name__ == '__main__':
    sys.exit(main())
