import chainer
import chainer.functions as F

from tgan_updater_base import TGANUpdaterBase


class TGANUpdaterLSGAN(TGANUpdaterBase):

    def update_core(self):
        xp = self.fsgen.xp
        fsgen_optimizer = self.get_optimizer('fsgen')
        vgen_optimizer = self.get_optimizer('vgen')
        vdis_optimizer = self.get_optimizer('vdis')
        real_video, fake_video, dis_fake, dis_real = self.forward()

        loss_dis_fake = F.mean_squared_error(dis_fake, xp.ones_like(dis_fake.data) * -1)
        loss_dis_real = F.mean_squared_error(dis_real, xp.ones_like(dis_real.data))
        loss_dis = loss_dis_fake + loss_dis_real
        loss_fsgen = F.mean_squared_error(dis_fake, xp.zeros_like(dis_fake.data))

        chainer.report({'loss_dis': loss_dis}, self.vdis)
        chainer.report({'loss_gen': loss_fsgen}, self.vdis)

        fsgen_optimizer.target.zerograds()
        vgen_optimizer.target.zerograds()
        vdis_optimizer.target.zerograds()

        loss_fsgen.backward()
        fsgen_optimizer.update()
        fake_video.unchain_backward()

        vgen_optimizer.update()
        loss_dis.backward()
        vdis_optimizer.update()
        self.it += 1
