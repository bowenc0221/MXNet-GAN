import mxnet as mx
import os
import numpy as np
import cv2

class pix2pixIter(mx.io.DataIter):
    def __init__(self, config, shuffle=False, ctx=None):
        self.config = config
        self.dataset = config.dataset.dataset  # name of dataset
        self.imageset = config.dataset.imageset  # name of image name text file
        self.root = os.path.join(config.dataset.root, config.dataset.dataset)  # path to store image name text file
        self.image_root = os.path.join(config.dataset.image_root, config.dataset.dataset)  # path to jpeg file

        self.image_files = self._load_image_path()
        self.size = len(self.image_files)
        self.index = np.arange(self.size)

        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]

        self.cur = 0
        self.shuffle = shuffle

        self.batch_size = config.TRAIN.BATCH_SIZE
        assert self.batch_size == 1

        self.AtoB = config.AtoB
        self.A = None
        self.B = None
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [('A', (1, 3, self.config.fineSize, self.config.fineSize)),
                ('B', (1, 3, self.config.fineSize, self.config.fineSize))]

    @property
    def provide_label(self):
        return []

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=[mx.nd.array(self.A, ctx=self.ctx),
                                         mx.nd.array(self.B, ctx=self.ctx)],
                                   label=self.getlabel(),
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def _load_image_path(self):
        fname = os.path.join(self.root, self.imageset + '.txt')
        assert os.path.exists(fname), 'Path does not exist: {}'.format(fname)
        with open(fname) as f:
            lines = [x.strip() for x in f.readlines()]

        return lines

    def get_batch(self):
        # cur_from = self.cur
        # cur_to = min(cur_from + self.batch_size, self.size)

        index = self.cur
        AB_path = os.path.join(self.image_root, self.image_files[index])
        AB = cv2.imread(AB_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        AB = cv2.resize(AB, (self.config.loadSize * 2, self.config.loadSize), interpolation = cv2.INTER_CUBIC)
        # AB = AB.resize((self.config.loadSize * 2, self.config.loadSize), Image.BICUBIC)  # size = (width, height)
        # AB = self.transform(AB)

        w_total = AB.shape[1]
        w = int(w_total / 2)
        h = AB.shape[0]
        w_offset = np.random.randint(0, max(0, w - self.config.fineSize - 1))
        h_offset = np.random.randint(0, max(0, h - self.config.fineSize - 1))

        # random crop
        A = AB[h_offset:h_offset + self.config.fineSize,
            w_offset:w_offset + self.config.fineSize, :]
        B = AB[h_offset:h_offset + self.config.fineSize,
            w + w_offset:w + w_offset + self.config.fineSize, :]

        if self.config.TRAIN.FLIP and np.random.random() < 0.5:
            A = A[:, ::-1, :]
            B = B[:, ::-1, :]

        # H x W x C -> H x W x C x 1 -> 1 x C x H x W
        A = np.transpose(A[..., np.newaxis], (3, 2, 0, 1))
        B = np.transpose(B[..., np.newaxis], (3, 2, 0, 1))

        if self.AtoB:
            self.A = A.astype(np.float32) / (255.0 / 2) - 1.0
            self.B = B.astype(np.float32) / (255.0 / 2) - 1.0
        else:
            self.B = A.astype(np.float32) / (255.0 / 2) - 1.0
            self.A = B.astype(np.float32) / (255.0 / 2) - 1.0
