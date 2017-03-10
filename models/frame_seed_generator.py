import chainer
import chainer.functions as F
import chainer.links as L


class FrameSeedGeneratorInitUniform(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, z_fast_dim=256, wscale=0.01):
        super(FrameSeedGeneratorInitUniform, self).__init__()
        w = chainer.initializers.Uniform(wscale)
        with self.init_scope():
            self.dc0 = L.DeconvolutionND(1, z_slow_dim, 512, 1, 1, 0, initialW=w)
            self.dc1 = L.DeconvolutionND(1, 512, 256, 4, 2, 1, initialW=w)
            self.dc2 = L.DeconvolutionND(1, 256, 128, 4, 2, 1, initialW=w)
            self.dc3 = L.DeconvolutionND(1, 128, 128, 4, 2, 1, initialW=w)
            self.dc4 = L.DeconvolutionND(1, 128, z_fast_dim, 4, 2, 1, initialW=w)
            self.bn0 = L.BatchNormalization(512)
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(128)
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim

    def __call__(self, z_slow):
        h = F.reshape(z_slow, (z_slow.shape[0], -1, 1))
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        z_fast = F.tanh(self.dc4(h))
        return z_fast


class FrameSeedGeneratorInitDefault(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, z_fast_dim=256):
        super(FrameSeedGeneratorInitDefault, self).__init__()
        w = None
        with self.init_scope():
            self.dc0 = L.DeconvolutionND(1, z_slow_dim, 512, 1, 1, 0, initialW=w)
            self.dc1 = L.DeconvolutionND(1, 512, 256, 4, 2, 1, initialW=w)
            self.dc2 = L.DeconvolutionND(1, 256, 128, 4, 2, 1, initialW=w)
            self.dc3 = L.DeconvolutionND(1, 128, 128, 4, 2, 1, initialW=w)
            self.dc4 = L.DeconvolutionND(1, 128, z_fast_dim, 4, 2, 1, initialW=w)
            self.bn0 = L.BatchNormalization(512)
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(128)
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim

    def __call__(self, z_slow):
        h = F.reshape(z_slow, (z_slow.shape[0], -1, 1))
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        z_fast = F.tanh(self.dc4(h))
        return z_fast


class FrameSeedGeneratorNoBetaInitUniform(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, z_fast_dim=256, wscale=0.01):
        super(FrameSeedGeneratorNoBetaInitUniform, self).__init__()
        w = chainer.initializers.Uniform(wscale)
        with self.init_scope():
            self.dc0 = L.DeconvolutionND(1, z_slow_dim, 512, 1, 1, 0, initialW=w)
            self.dc1 = L.DeconvolutionND(1, 512, 256, 4, 2, 1, initialW=w)
            self.dc2 = L.DeconvolutionND(1, 256, 128, 4, 2, 1, initialW=w)
            self.dc3 = L.DeconvolutionND(1, 128, 128, 4, 2, 1, initialW=w)
            self.dc4 = L.DeconvolutionND(1, 128, z_fast_dim, 4, 2, 1, initialW=w)
            self.bn0 = L.BatchNormalization(512, use_beta=False)
            self.bn1 = L.BatchNormalization(256, use_beta=False)
            self.bn2 = L.BatchNormalization(128, use_beta=False)
            self.bn3 = L.BatchNormalization(128, use_beta=False)
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim

    def __call__(self, z_slow):
        h = F.reshape(z_slow, (z_slow.shape[0], -1, 1))
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        z_fast = F.tanh(self.dc4(h))
        return z_fast


class FrameSeedGeneratorNoBetaInitDefault(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, z_fast_dim=256):
        super(FrameSeedGeneratorNoBetaInitDefault, self).__init__()
        w = None
        with self.init_scope():
            self.dc0 = L.DeconvolutionND(1, z_slow_dim, 512, 1, 1, 0, initialW=w)
            self.dc1 = L.DeconvolutionND(1, 512, 256, 4, 2, 1, initialW=w)
            self.dc2 = L.DeconvolutionND(1, 256, 128, 4, 2, 1, initialW=w)
            self.dc3 = L.DeconvolutionND(1, 128, 128, 4, 2, 1, initialW=w)
            self.dc4 = L.DeconvolutionND(1, 128, z_fast_dim, 4, 2, 1, initialW=w)
            self.bn0 = L.BatchNormalization(512, use_beta=False)
            self.bn1 = L.BatchNormalization(256, use_beta=False)
            self.bn2 = L.BatchNormalization(128, use_beta=False)
            self.bn3 = L.BatchNormalization(128, use_beta=False)
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim

    def __call__(self, z_slow):
        h = F.reshape(z_slow, (z_slow.shape[0], -1, 1))
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        z_fast = F.tanh(self.dc4(h))
        return z_fast
