import chainer
import chainer.functions as F
import numpy as np
from chainer import Variable


class DCGANUpdaterBase(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.it = 0
        self.fsgen, self.vgen, self.vdis = kwargs.pop('models')
        self.shuffle_time_order = kwargs.pop('shuffle_time_order') if 'shuffle_time_order' in kwargs else None
        super(DCGANUpdaterBase, self).__init__(*args, **kwargs)

    def _get_real_video(self):
        xp = self.fsgen.xp
        batch = self.get_iterator('main').next()
        batch = xp.array(batch, dtype=np.float32).transpose(0, 2, 1, 3, 4)
        return Variable(batch)

    def _get_seeds(self, batchsize):
        xp = self.fsgen.xp
        z_slow = Variable(
            xp.random.uniform(
                -1, 1, (batchsize, self.fsgen.z_slow_dim)).astype(np.float32))
        with chainer.using_config('train', True):
            z_fast = self.fsgen(z_slow)
        return z_slow, z_fast

    def _generate_fake_video(self, z_slow, z_fast):
        n_b, z_fast_dim, n_frames = z_fast.shape
        self.batchsize = n_b
        z_fast = F.reshape(F.transpose(
            z_fast, [0, 2, 1]), (n_b * n_frames, z_fast_dim))

        n_b, z_slow_dim = z_slow.shape
        z_slow = F.reshape(
            F.broadcast_to(F.reshape(z_slow, (n_b, 1, z_slow_dim)),
                           (n_b, n_frames, z_slow_dim)),
            (n_b * n_frames, z_slow_dim))

        with chainer.using_config('train', True):
            fake_video = self.vgen(z_slow, z_fast)
            _, n_ch, h, w = fake_video.shape

        fake_video = F.transpose(
            F.reshape(fake_video, (n_b, n_frames, n_ch, h, w)),
            [0, 2, 1, 3, 4])
        return fake_video

    def _shuffle_time_order(self, video):
        frame_order = np.arange(video.shape[2])
        np.random.shuffle(frame_order)
        return video[:, :, frame_order.tolist(), :, :]

    def forward(self):
        real_video = self._get_real_video()
        batchsize = real_video.shape[0]
        z_slow, z_fast = self._get_seeds(batchsize)
        fake_video = self._generate_fake_video(z_slow, z_fast)

        # dis_out = self.vdis(F.concat([fake_video, real_video], axis=2))
        # dis_fake, dis_real = F.split_axis(dis_out, 2, 0)
        if self.shuffle_time_order is not None:
            if np.random.rand() > self.shuffle_time_order:
                fake_video = self._shuffle_time_order(fake_video)
                real_video = self._shuffle_time_order(real_video)
        dis_fake = self.vdis(fake_video)
        dis_real = self.vdis(real_video)

        return real_video, fake_video, dis_fake, dis_real
