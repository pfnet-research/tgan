import chainer
import chainer.functions as F
import numpy as np
from scipy import linalg
from multiprocessing import Pool

from tgan_updater_base import TGANUpdaterBase


class TGANUpdaterVanilla(TGANUpdaterBase):

    def update_core(self):
        xp = self.fsgen.xp
        fsgen_optimizer = self.get_optimizer('fsgen')
        vgen_optimizer = self.get_optimizer('vgen')
        vdis_optimizer = self.get_optimizer('vdis')
        real_video, fake_video, dis_fake, dis_real = self.forward()
        batchsize = real_video.shape[0]
        
        loss_dis_fake = F.sigmoid_cross_entropy(
            dis_fake, xp.ones((batchsize, 1, 1, 1), dtype="i"))
        loss_dis_real = F.sigmoid_cross_entropy(
            dis_real, xp.zeros((batchsize, 1, 1, 1), dtype="i"))
        loss_gen = F.sigmoid_cross_entropy(
            dis_fake, xp.zeros((batchsize, 1, 1, 1), dtype="i"))
        
        chainer.report({'loss_dis_fake': loss_dis_fake}, self.vdis)
        chainer.report({'loss_dis_real': loss_dis_real}, self.vdis)

        fsgen_optimizer.target.zerograds()
        vgen_optimizer.target.zerograds()
        loss_gen.backward()
        fsgen_optimizer.update()
        vgen_optimizer.update()
        fake_video.unchain_backward()

        vdis_optimizer.target.zerograds()
        (loss_dis_fake + loss_dis_real).backward()
        vdis_optimizer.update()
