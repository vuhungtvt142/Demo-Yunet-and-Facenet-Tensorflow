import numpy as np
import onnxruntime
from PIL import Image


def process(img, loc):
    img_size = np.asarray(img.shape)[0:2]
    (t, r, b, l) = loc

    l32 = np.maximum(l-16, 0)
    t32 = np.maximum(t-16, 0)
    r32 = np.minimum(r+16, img_size[1])
    b32 = np.minimum(b+16, img_size[0])

    cropped = img[t32:b32, l32:r32, :]
    aligned = np.array(Image.fromarray(cropped).resize(
        (160, 160), resample=Image.BILINEAR))
    std = (aligned.astype(np.float32) - 127.5)/128.0

    return std


class Encoder():
    def __init__(self):
        self.session = onnxruntime.InferenceSession('models/20180402-114759-optimize.onnx')

    def encode(self, img, locs):
        if len(locs) == 0:
            return []

        images = np.stack([process(img, loc) for loc in locs])

        # Run forward pass to calculate embeddings
        emb = self.session.run(['embeddings:0'], {'input:0': images})[0]

        return emb

    def _raw_encode(self, images):
        return self.session.run(['embeddings:0'], {'input:0': images})[0]
