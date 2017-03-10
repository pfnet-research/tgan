import chainer
import chainer.functions as F
import numpy as np
from scipy import linalg
from multiprocessing import Pool

from tgan_updater_base import TGANUpdaterBase


def _clip_singular_value(A, name=None):
    U, s, Vh = linalg.svd(A, full_matrices=False)
    s[s > 1] = 1
    if name:
        return name, np.dot(np.dot(U, np.diag(s)), Vh)
    else:
        return np.dot(np.dot(U, np.diag(s)), Vh)


class TGANUpdaterWGANSVC(TGANUpdaterBase):

    def __init__(self, *args, **kwargs):
        self.freq = kwargs.pop('freq') if 'freq' in kwargs else 1
        super(TGANUpdaterWGANSVC, self).__init__(*args, **kwargs)

    def update_core(self):
        xp = self.fsgen.xp
        fsgen_optimizer = self.get_optimizer('fsgen')
        vgen_optimizer = self.get_optimizer('vgen')
        vdis_optimizer = self.get_optimizer('vdis')
        real_video, fake_video, dis_fake, dis_real = self.forward()

        loss_dis_fake = F.sum(dis_fake) / self.batchsize
        loss_dis_real = F.sum(-dis_real) / self.batchsize
        loss_gen = F.sum(-dis_fake) / self.batchsize
        loss_dis = loss_dis_real + loss_dis_fake

        chainer.report({'loss_gen': loss_gen}, self.vdis)
        chainer.report({'loss_dis': loss_dis}, self.vdis)

        fsgen_optimizer.target.zerograds()
        vgen_optimizer.target.zerograds()
        loss_gen.backward()
        fsgen_optimizer.update()
        vgen_optimizer.update()
        fake_video.unchain_backward()

        vdis_optimizer.target.zerograds()
        loss_dis.backward()
        vdis_optimizer.update()
        self.it += 1

        if (self.it % self.freq) == 0:
            for p in self.vdis.params():
                if p.data.ndim >= 4:
                    A = p.data.reshape((p.data.shape[0], -1)).get()
                    A = _clip_singular_value(A)
                    p.data = xp.asarray(A.reshape(p.data.shape))

            for n in self.vdis.links():
                if 'bn' in str(n.name):
                    gamma = n.gamma.data.get()
                    std = np.sqrt(n.avg_var.get())
                    gamma[gamma > std] = std[gamma > std]
                    gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
                    n.gamma.data = xp.asarray(gamma)


class TGANUpdaterWGANParallelSVC(TGANUpdaterBase):

    def __init__(self, *args, **kwargs):
        self.freq = kwargs.pop('freq') if 'freq' in kwargs else 1
        super(TGANUpdaterWGANParallelSVC, self).__init__(*args, **kwargs)
        self.p = Pool()

    def update_core(self):
        xp = self.fsgen.xp
        fsgen_optimizer = self.get_optimizer('fsgen')
        vgen_optimizer = self.get_optimizer('vgen')
        vdis_optimizer = self.get_optimizer('vdis')
        real_video, fake_video, dis_fake, dis_real = self.forward()

        loss_dis_fake = F.sum(dis_fake) / self.batchsize
        loss_dis_real = F.sum(-dis_real) / self.batchsize
        loss_gen = F.sum(-dis_fake) / self.batchsize
        loss_dis = loss_dis_real + loss_dis_fake

        chainer.report({'loss_gen': loss_gen}, self.vdis)
        chainer.report({'loss_dis': loss_dis}, self.vdis)

        fsgen_optimizer.target.zerograds()
        vgen_optimizer.target.zerograds()
        loss_gen.backward()
        fsgen_optimizer.update()
        vgen_optimizer.update()
        fake_video.unchain_backward()

        vdis_optimizer.target.zerograds()
        loss_dis.backward()
        vdis_optimizer.update()
        self.it += 1

        if (self.it % self.freq) == 0:
            namedparams = dict(self.vdis.namedparams())
            ret = [self.p.apply_async(
                _clip_singular_value,
                (p.data.reshape(p.data.shape[0], -1).get(), name))
                for name, p in namedparams.items() if p.data.ndim >= 4]
            for r in ret:
                name, A = r.get()
                p = namedparams[name]
                p.data = xp.asarray(A.reshape(p.data.shape))

            for n in self.vdis.links():
                if 'bn' in str(n.name):
                    gamma = n.gamma.data.get()
                    std = np.sqrt(n.avg_var.get())
                    gamma[gamma > std] = std[gamma > std]
                    gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
                    n.gamma.data = xp.asarray(gamma)


class TGANUpdaterWGANVideoFrameDis(TGANUpdaterBase):

    def __init__(self, *args, **kwargs):
        self.freq = kwargs.pop('freq') if 'freq' in kwargs else 1
        super(TGANUpdaterWGANVideoFrameDis, self).__init__(*args, **kwargs)

    def update_core(self):
        xp = self.fsgen.xp
        fsgen_optimizer = self.get_optimizer('fsgen')
        vgen_optimizer = self.get_optimizer('vgen')
        vdis_optimizer = self.get_optimizer('vdis')
        real_video, fake_video, dis_fake, dis_real = self.forward()

        vid_loss_dis = (F.sum(dis_fake[0]) + F.sum(-dis_real[0])) / self.batchsize
        frame_loss_dis = (F.sum(dis_fake[1]) + F.sum(-dis_real[1])) / self.batchsize
        loss_dis = vid_loss_dis + frame_loss_dis
        vid_loss_gen = F.sum(-dis_fake[0]) / self.batchsize
        frame_loss_gen = F.sum(-dis_fake[1]) / self.batchsize
        loss_gen = vid_loss_gen + frame_loss_gen

        chainer.report({'loss_gen': loss_gen}, self.vdis)
        chainer.report({'loss_dis': loss_dis}, self.vdis)

        fsgen_optimizer.target.zerograds()
        vgen_optimizer.target.zerograds()
        loss_gen.backward()
        fsgen_optimizer.update()
        vgen_optimizer.update()
        fake_video.unchain_backward()

        vdis_optimizer.target.zerograds()
        loss_dis.backward()
        vdis_optimizer.update()
        self.it += 1

        if (self.it % self.freq) == 0:
            for p in self.vdis.params():
                if p.data.ndim >= 4:
                    A = p.data.reshape((p.data.shape[0], -1)).get()
                    A = _clip_singular_value(A)
                    p.data = xp.asarray(A.reshape(p.data.shape))

            for n in self.vdis.links():
                if 'bn' in str(n.name):
                    gamma = n.gamma.data.get()
                    std = np.sqrt(n.avg_var.get())
                    gamma[gamma > std] = std[gamma > std]
                    gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
                    n.gamma.data = xp.asarray(gamma)


class TGANUpdaterWGANClip(TGANUpdaterBase):

    def update_core(self):
        xp = self.fsgen.xp
        fsgen_optimizer = self.get_optimizer('fsgen')
        vgen_optimizer = self.get_optimizer('vgen')
        vdis_optimizer = self.get_optimizer('vdis')
        real_video, fake_video, dis_fake, dis_real = self.forward()

        loss_dis_fake = F.sum(dis_fake) / self.batchsize
        loss_dis_real = F.sum(-dis_real) / self.batchsize
        loss_gen = F.sum(-dis_fake) / self.batchsize
        loss_dis = loss_dis_real + loss_dis_fake

        chainer.report({'loss_gen': loss_gen}, self.vdis)
        chainer.report({'loss_dis': loss_dis}, self.vdis)

        fsgen_optimizer.target.zerograds()
        vgen_optimizer.target.zerograds()
        loss_gen.backward()
        fsgen_optimizer.update()
        vgen_optimizer.update()
        fake_video.unchain_backward()

        vdis_optimizer.target.zerograds()
        loss_dis.backward()
        vdis_optimizer.update()
        self.it += 1

        for p in self.vdis.params():
            p.data = xp.clip(p.data, -0.01, 0.01)
