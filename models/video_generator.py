import chainer
import chainer.functions as F
import chainer.links as L


class VideoGeneratorInitUniform(chainer.Chain):

    def __init__(self, z_slow_dim, z_fast_dim, out_channels, bottom_width,
                 conv_ch=512, wscale=0.01):
        self.ch = conv_ch
        self.bottom_width = bottom_width
        slow_mid_dim = bottom_width * bottom_width * conv_ch // 2
        fast_mid_dim = bottom_width * bottom_width * conv_ch // 2
        super(VideoGeneratorInitUniform, self).__init__()
        w = chainer.initializers.Uniform(wscale)
        with self.init_scope():
            self.l0s = L.Linear(z_slow_dim, slow_mid_dim, initialW=w, nobias=True)
            self.l0f = L.Linear(z_fast_dim, fast_mid_dim, initialW=w, nobias=True)
            self.dc1 = L.Deconvolution2D(conv_ch, conv_ch // 2, 4, 2, 1, initialW=w, nobias=True)
            self.dc2 = L.Deconvolution2D(conv_ch // 2, conv_ch // 4, 4, 2, 1, initialW=w, nobias=True)
            self.dc3 = L.Deconvolution2D(conv_ch // 4, conv_ch // 8, 4, 2, 1, initialW=w, nobias=True)
            self.dc4 = L.Deconvolution2D(conv_ch // 8, conv_ch // 16, 4, 2, 1, initialW=w, nobias=True)
            self.dc5 = L.Deconvolution2D(conv_ch // 16, out_channels, 3, 1, 1, initialW=w, nobias=False)
            self.bn0s = L.BatchNormalization(slow_mid_dim)
            self.bn0f = L.BatchNormalization(fast_mid_dim)
            self.bn1 = L.BatchNormalization(conv_ch // 2)
            self.bn2 = L.BatchNormalization(conv_ch // 4)
            self.bn3 = L.BatchNormalization(conv_ch // 8)
            self.bn4 = L.BatchNormalization(conv_ch // 16)

    def __call__(self, z_slow, z_fast):
        n = z_slow.shape[0]
        h_slow = F.reshape(F.relu(self.bn0s(self.l0s(z_slow))),
                           (n, self.ch // 2, self.bottom_width, self.bottom_width))
        h_fast = F.reshape(F.relu(self.bn0f(self.l0f(z_fast))),
                           (n, self.ch // 2, self.bottom_width, self.bottom_width))
        h = F.concat([h_slow, h_fast], axis=1)
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        x = F.tanh(self.dc5(h))
        return x


class VideoGeneratorInitDefault(chainer.Chain):

    def __init__(self, z_slow_dim, z_fast_dim, out_channels, bottom_width,
                 conv_ch=512, wscale=0.02):
        self.ch = conv_ch
        self.bottom_width = bottom_width
        slow_mid_dim = bottom_width * bottom_width * conv_ch // 2
        fast_mid_dim = bottom_width * bottom_width * conv_ch // 2
        super(VideoGeneratorInitDefault, self).__init__()
        w = None
        with self.init_scope():
            self.l0s = L.Linear(z_slow_dim, slow_mid_dim, initialW=w, nobias=True)
            self.l0f = L.Linear(z_fast_dim, fast_mid_dim, initialW=w, nobias=True)
            self.dc1 = L.Deconvolution2D(conv_ch, conv_ch // 2, 4, 2, 1, initialW=w, nobias=True)
            self.dc2 = L.Deconvolution2D(conv_ch // 2, conv_ch // 4, 4, 2, 1, initialW=w, nobias=True)
            self.dc3 = L.Deconvolution2D(conv_ch // 4, conv_ch // 8, 4, 2, 1, initialW=w, nobias=True)
            self.dc4 = L.Deconvolution2D(conv_ch // 8, conv_ch // 16, 4, 2, 1, initialW=w, nobias=True)
            self.dc5 = L.Deconvolution2D(conv_ch // 16, out_channels, 3, 1, 1, initialW=w, nobias=False)
            self.bn0s = L.BatchNormalization(slow_mid_dim)
            self.bn0f = L.BatchNormalization(fast_mid_dim)
            self.bn1 = L.BatchNormalization(conv_ch // 2)
            self.bn2 = L.BatchNormalization(conv_ch // 4)
            self.bn3 = L.BatchNormalization(conv_ch // 8)
            self.bn4 = L.BatchNormalization(conv_ch // 16)

    def __call__(self, z_slow, z_fast):
        n = z_slow.shape[0]
        h_slow = F.reshape(F.relu(self.bn0s(self.l0s(z_slow))),
                           (n, self.ch // 2, self.bottom_width, self.bottom_width))
        h_fast = F.reshape(F.relu(self.bn0f(self.l0f(z_fast))),
                           (n, self.ch // 2, self.bottom_width, self.bottom_width))
        h = F.concat([h_slow, h_fast], axis=1)
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        x = F.tanh(self.dc5(h))
        return x
