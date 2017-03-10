import chainer
import chainer.functions as F
import chainer.links as L


class VideoDiscriminatorInitUniform(chainer.Chain):

    def __init__(self, in_channels, top_width, mid_ch, wscale=0.01):
        super(VideoDiscriminatorInitUniform, self).__init__()
        w = chainer.initializers.Uniform(wscale)
        with self.init_scope():
            self.c0 = L.ConvolutionND(3, in_channels, mid_ch, 4, 2, 1, initialW=w)
            self.c1 = L.ConvolutionND(3, mid_ch, mid_ch * 2, 4, 2, 1, initialW=w)
            self.c2 = L.ConvolutionND(3, mid_ch * 2, mid_ch * 4, 4, 2, 1, initialW=w)
            self.c3 = L.ConvolutionND(3, mid_ch * 4, mid_ch * 8, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(mid_ch * 8, 1, top_width, 1, 0, initialW=w)
            self.bn0 = L.BatchNormalization(mid_ch)
            self.bn1 = L.BatchNormalization(mid_ch * 2)
            self.bn2 = L.BatchNormalization(mid_ch * 4)
            self.bn3 = L.BatchNormalization(mid_ch * 8)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = F.reshape(h, (h.shape[0] * h.shape[2],) + self.c4.W.shape[1:])
        h = self.c4(h)
        return h


class VideoDiscriminatorInitDefault(chainer.Chain):

    def __init__(self, in_channels, top_width, mid_ch):
        super(VideoDiscriminatorInitDefault, self).__init__()
        w = None
        with self.init_scope():
            self.c0 = L.ConvolutionND(3, in_channels, mid_ch, 4, 2, 1, initialW=w)
            self.c1 = L.ConvolutionND(3, mid_ch, mid_ch * 2, 4, 2, 1, initialW=w)
            self.c2 = L.ConvolutionND(3, mid_ch * 2, mid_ch * 4, 4, 2, 1, initialW=w)
            self.c3 = L.ConvolutionND(3, mid_ch * 4, mid_ch * 8, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(mid_ch * 8, 1, top_width, 1, 0, initialW=w)
            self.bn0 = L.BatchNormalization(mid_ch)
            self.bn1 = L.BatchNormalization(mid_ch * 2)
            self.bn2 = L.BatchNormalization(mid_ch * 4)
            self.bn3 = L.BatchNormalization(mid_ch * 8)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = F.reshape(h, (h.shape[0] * h.shape[2],) + self.c4.W.shape[1:])
        h = self.c4(h)
        return h


class VideoDiscriminatorNoBetaInitUniform(chainer.Chain):

    def __init__(self, in_channels, top_width, mid_ch, wscale=0.01):
        super(VideoDiscriminatorNoBetaInitUniform, self).__init__()
        w = chainer.initializers.Uniform(wscale)
        with self.init_scope():
            self.c0 = L.ConvolutionND(3, in_channels, mid_ch, 4, 2, 1, initialW=w)
            self.c1 = L.ConvolutionND(3, mid_ch, mid_ch * 2, 4, 2, 1, initialW=w)
            self.c2 = L.ConvolutionND(3, mid_ch * 2, mid_ch * 4, 4, 2, 1, initialW=w)
            self.c3 = L.ConvolutionND(3, mid_ch * 4, mid_ch * 8, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(mid_ch * 8, 1, top_width, 1, 0, initialW=w)
            self.bn0 = L.BatchNormalization(mid_ch, use_beta=False)
            self.bn1 = L.BatchNormalization(mid_ch * 2, use_beta=False)
            self.bn2 = L.BatchNormalization(mid_ch * 4, use_beta=False)
            self.bn3 = L.BatchNormalization(mid_ch * 8, use_beta=False)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = F.reshape(h, (h.shape[0] * h.shape[2],) + self.c4.W.shape[1:])
        h = self.c4(h)
        return h


class VideoDiscriminatorNoBetaInitDefault(chainer.Chain):

    def __init__(self, in_channels, top_width, mid_ch):
        super(VideoDiscriminatorNoBetaInitDefault, self).__init__()
        w = None
        with self.init_scope():
            self.c0 = L.ConvolutionND(3, in_channels, mid_ch, 4, 2, 1, initialW=w)
            self.c1 = L.ConvolutionND(3, mid_ch, mid_ch * 2, 4, 2, 1, initialW=w)
            self.c2 = L.ConvolutionND(3, mid_ch * 2, mid_ch * 4, 4, 2, 1, initialW=w)
            self.c3 = L.ConvolutionND(3, mid_ch * 4, mid_ch * 8, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(mid_ch * 8, 1, top_width, 1, 0, initialW=w)
            self.bn0 = L.BatchNormalization(mid_ch, use_beta=False)
            self.bn1 = L.BatchNormalization(mid_ch * 2, use_beta=False)
            self.bn2 = L.BatchNormalization(mid_ch * 4, use_beta=False)
            self.bn3 = L.BatchNormalization(mid_ch * 8, use_beta=False)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = F.reshape(h, (h.shape[0] * h.shape[2],) + self.c4.W.shape[1:])
        h = self.c4(h)
        return h


def add_noise(h, sigma=0.3):
    xp = chainer.cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.data.shape)
    else:
        return h


class VideoDiscriminatorNoBetaInitDefaultWithNoise(chainer.Chain):

    def __init__(self, in_channels, top_width, mid_ch, sigma):
        super(VideoDiscriminatorNoBetaInitDefaultWithNoise, self).__init__()
        w = None
        with self.init_scope():
            self.c0 = L.ConvolutionND(3, in_channels, mid_ch, 4, 2, 1, initialW=w)
            self.c1 = L.ConvolutionND(3, mid_ch, mid_ch * 2, 4, 2, 1, initialW=w)
            self.c2 = L.ConvolutionND(3, mid_ch * 2, mid_ch * 4, 4, 2, 1, initialW=w)
            self.c3 = L.ConvolutionND(3, mid_ch * 4, mid_ch * 8, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(mid_ch * 8, 1, top_width, 1, 0, initialW=w)
            self.bn0 = L.BatchNormalization(mid_ch, use_beta=False)
            self.bn1 = L.BatchNormalization(mid_ch * 2, use_beta=False)
            self.bn2 = L.BatchNormalization(mid_ch * 4, use_beta=False)
            self.bn3 = L.BatchNormalization(mid_ch * 8, use_beta=False)
        self.sigma = sigma

    def __call__(self, x):
        h = F.leaky_relu(add_noise(self.c0(x), sigma=self.sigma))
        h = self.bn1(self.c1(h))
        h = F.leaky_relu(add_noise(h, sigma=self.sigma))
        h = self.bn2(self.c2(h))
        h = F.leaky_relu(add_noise(h, sigma=self.sigma))
        h = self.bn3(self.c3(h))
        h = F.leaky_relu(add_noise(h, sigma=self.sigma))
        h = F.reshape(h, (h.shape[0] * h.shape[2],) + self.c4.W.shape[1:])
        h = self.c4(h)
        return h


class FrameDiscriminator(chainer.Chain):

    def __init__(self, mid_ch=64):
        super(FrameDiscriminator, self).__init__()
        w = None
        with self.init_scope():
            self.c0 = L.Convolution2D(None, mid_ch, 4, 2, 1, initialW=w)
            self.c1 = L.Convolution2D(mid_ch, mid_ch * 2, 4, 2, 1, initialW=w)
            self.c2 = L.Convolution2D(mid_ch * 2, mid_ch * 4, 4, 2, 1, initialW=w)
            self.c3 = L.Convolution2D(mid_ch * 4, mid_ch * 8, 4, 2, 1, initialW=w)
            self.l4l = L.Linear(None, 1, initialW=w)
            self.bn0 = L.BatchNormalization(64)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return l


class VideoAndFrameDiscriminator(chainer.Chain):

    def __init__(self, in_channels, top_width, mid_ch):
        super(VideoAndFrameDiscriminator, self).__init__()
        with self.init_scope():
            self.frame_dis = FrameDiscriminator(mid_ch)
            self.video_dis = VideoDiscriminator(in_channels, top_width, mid_ch)

    def __call__(self, img):
        video_y = self.video_dis(img)
        n, c, nf, h, w = img.shape
        img = F.reshape(img, (n, c * nf, h, w))
        frame_y = self.frame_dis(img)
        return video_y, frame_y
