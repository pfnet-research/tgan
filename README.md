Temporal Generative Adversarial Nets
====================================

**The new version of TGAN has been published and the code is available: [TGANv2](https://github.com/pfnet-research/tgan2).**

This repository contains a collection of scripts used in the experiments of
[Temporal Generative Adversarial Nets with Singular Value Clipping](https://arxiv.org/abs/1611.06624).

Disclaimer: PFN provides no warranty or support for this implementation. Use it at your own risk. See [license](LICENSE.md) for details.

## Results

![](https://raw.githubusercontent.com/wiki/pfnet-research/tgan/images/ucf_cond_scaled.gif)

## Requirements

These scripts require the following python libraries.

- Chainer 2.0.0+
- h5py
- numpy
- pandas
- PIL
- PyYAML
- matplotlib

Note that they also require ffmpeg to produce a video from a set of images.

## Usage

### Datasets

In order to run our scripts, you need to prepare MovingMNIST and UCF-101 datasets as follows.

#### MovingMNIST

1. Download `mnist_test_seq.npy` from [here](http://www.cs.toronto.edu/~nitish/unsupervised_video/).
2. Put it on `path-to-tgans/data/mnist_test_seq.npy`.

#### UCF-101

There are two ways to create an UCF-101 dataset for this script.

1. Transforms all the videos in the UCF-101 dataset to the images.
2. Resizes these images to the appropriate resolution, and concatenate
   them into as single hdf5 format represented as (time, channel, rows, cols).
   In this transformation we used ``make_ucf101.py`` in this repository.
   Note that this script also produces a config file that describes videos and
   these corresponding label information.
3. puts them on `path-to-tgans/data`.

Another way is to simply download these files; please download them from
[this url](https://www.dropbox.com/sh/j9fsakeuvicpeo8/AAD6BVhbZRyi7NXaMfn6TO4da?dl=0),
and put them on the same directory.

### Training

#### TGAN with WGAN and Singular Value Clipping

```
python train.py --config_path configs/moving_mnist/mnist_wgan_svd_zdim-100_no-beta-all_init-uniform-all.yml --gpu 0
python train.py --config_path configs/ucf101/ucf101_wgan_svd_zdim-100_no-beta.yml --gpu 0
```

#### TGAN (WGAN and weight clipping)

```
python train.py --config_path configs/moving_mnist/mnist_wgan_clip_zdim-100_no-beta-all_init-uniform-all.yml --gpu 0
python train.py --config_path configs/ucf101/ucf101_wgan_clip_zdim-100_no-beta.yml --gpu 0
```

#### TGAN (vanilla GAN)

```
python train.py --config_path configs/ucf101/ucf101_vanilla_zdim-100_no-beta.yml --gpu 0
```

## Quantitative evaluation on UCF101 (2019/08/20)

We have uploaded ``mean2.npz`` on GitHub because there are many inquiries about the mean file in the UCF101.
If you want to perform a quantitative evaluation, please download it from
[this url](https://github.com/pfnet-research/tgan/releases/download/v1.0.0/mean2.npz).

## Citation

Please cite the paper if you are interested in:

```
@inproceedings{TGAN2017,
    author = {Saito, Masaki and Matsumoto, Eiichi and Saito, Shunta},
    title = {Temporal Generative Adversarial Nets with Singular Value Clipping},
    booktitle = {ICCV},
    year = {2017},
}
```

## License

MIT License. Please see the LICENSE file for details.
