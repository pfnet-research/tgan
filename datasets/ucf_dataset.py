import chainer
import h5py
import numpy as np
import pandas as pd


class UCFDataset(chainer.dataset.DatasetMixin):

    def __init__(self, n_frames, h5path, config_path, img_size):
        self.h5file = h5py.File(h5path, 'r')
        self.dset = self.h5file['image']
        self.conf = pd.read_pickle(config_path)
        self.ind = self.conf.index.tolist()
        self.n_frames = n_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.conf)

    def _crop_center(self, x):
        x = x[:, :, :, 10:10 + self.img_size]
        # x = x[:, :, x.shape[2] // 2 - self.img_size // 2:
        #       x.shape[2] // 2 + self.img_size // 2,
        #       x.shape[3] // 2 - self.img_size // 2:
        #       x.shape[3] // 2 + self.img_size // 2]
        assert x.shape[2] == self.img_size
        assert x.shape[3] == self.img_size
        return x

    def get_example(self, i):
        mov_info = self.conf.loc[self.ind[i]]
        length = mov_info.end - mov_info.start
        offset = np.random.randint(length - self.n_frames) \
            if length > self.n_frames else 0
        x = self.dset[mov_info.start + offset:
                      mov_info.start + offset + self.n_frames]
        x = self._crop_center(x)
        return np.asarray((x - 128.0) / 128.0, dtype=np.float32)


class UCFDatasetAugmented(chainer.dataset.DatasetMixin):

    def __init__(self, n_frames, h5path, config_path, img_size):
        self.h5file = h5py.File(h5path, 'r')
        self.dset = self.h5file['image']
        self.conf = pd.read_pickle(config_path)
        self.ind = self.conf.index.tolist()
        self.n_frames = n_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.conf)

    def _crop_center(self, x):
        h_shift = np.random.randint(x.shape[2] - self.img_size) if x.shape[2] > self.img_size else 0
        w_shift = np.random.randint(x.shape[3] - self.img_size) if x.shape[3] > self.img_size else 0
        x = x[:, :,
              h_shift:h_shift + self.img_size,
              w_shift:w_shift + self.img_size]
        assert x.shape[2] == self.img_size
        assert x.shape[3] == self.img_size
        return x

    def get_example(self, i):
        mov_info = self.conf.loc[self.ind[i]]
        length = mov_info.end - mov_info.start
        offset = np.random.randint(length - self.n_frames) \
            if length > self.n_frames else 0
        x = self.dset[mov_info.start + offset:
                      mov_info.start + offset + self.n_frames]
        x = self._crop_center(x)
        if np.random.rand() > 0.5:
            x = x[:, :, :, ::-1]
        return np.asarray((x - 128.0) / 128.0, dtype=np.float32)
