import chainer
import numpy as np


class MovingMNISTDataset(chainer.dataset.DatasetMixin):

    def __init__(self, n_frames, dataset_path):
        self.dset = np.load(dataset_path)
        self.dset = self.dset.transpose([1, 0, 2, 3])
        self.dset = self.dset[:, :, np.newaxis, :, :]
        self.n_frames = n_frames

    def __len__(self):
        return self.dset.shape[0]

    def get_example(self, i):
        T = self.dset.shape[1]
        ot = np.random.randint(T - self.n_frames) if T > self.n_frames else 0
        x = self.dset[i, ot:(ot + self.n_frames)]
        return np.asarray((x - 128.0) / 128.0, dtype=np.float32)
