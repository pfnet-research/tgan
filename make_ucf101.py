import logging
import os
import re
import sys
import pickle
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from PIL import Image


def init_logger(filename=''):
    kwargs = {
        'format': "%(asctime)s [%(levelname)s] %(message)s",
        'level': logging.INFO,
    }
    if filename != '':
        kwargs['filename'] = filename
    logging.basicConfig(**kwargs)
    return logging.getLogger(__name__)


def main():
    logger = init_logger()

    img_dir = 'ucf101/images/test'
    dst_path = 'ucf101/64px/ucf101_test.h5'
    dst_config = 'ucf101/64px/ucf101_test_conf.pkl'
    dst_config_pd = 'ucf101/64px/ucf101_test_conf_pd.pkl'
    rows, cols = 64, 85

    img_path = os.path.join('dataset', img_dir)
    images = os.listdir(img_path)

    time_dict = defaultdict(int)
    for filename in images:
        result = re.search(r'v_(?P<video>.+)_(?P<time>\d+).png', filename)
        assert(result is not None)
        video = result.group('video')
        t = int(result.group('time')) + 1
        if time_dict[video] < t:
            time_dict[video] = t

    videos = sorted(list(time_dict.items()))
    logger.info('# of videos: %i', len(videos))
    n_frames = sum(t for (v, t) in videos)
    logger.info('# of frames: %i', n_frames)

    logger.info('Making h5file...')
    h5file = h5py.File(os.path.join('dataset', dst_path), 'w')

    # (time, channel, rows, cols)
    shape = (n_frames, 3, rows, cols)
    dset = h5file.create_dataset('image', shape, dtype='u1')

    config = []
    config_pd = []
    ts = 0
    for video, time in videos:
        logger.info('Now processing %s...', video)
        for t in range(time):
            i = ts + t
            filepath = os.path.join(img_path, 'v_{}_{}.png'.format(video, t))
            img = np.asarray(
                Image.open(filepath).resize((cols, rows)),
                dtype=np.uint8).transpose(2, 0, 1)
            dset[i] = img
        config.append((ts, ts + time))
        result = re.search(
            r'(?P<category>.+)_g(?P<group>\d+)_c(?P<scene>\d+)', video)
        assert(result is not None)
        config_pd.append({
            'category': result.group('category'),
            'group': int(result.group('group')),
            'scene': int(result.group('scene')),
            'start': ts, 'end': ts + time,
        })
        ts = ts + time

    with open(os.path.join('dataset', dst_config), 'wb') as fp:
        pickle.dump(config, fp)

    config_frame = pd.DataFrame(config_pd)
    config_frame.to_pickle(os.path.join('dataset', dst_config_pd))

    logger.info('Done.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
