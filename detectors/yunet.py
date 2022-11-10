from itertools import product as product

import cv2
import numpy as np
import onnxruntime


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def nms(dets, thresh, opencv_mode=False):
    if opencv_mode:
        _boxes = dets[:, :4].copy()
        scores = dets[:, -1]
        _boxes[:, 2] = _boxes[:, 2] - _boxes[:, 0]
        _boxes[:, 3] = _boxes[:, 3] - _boxes[:, 1]
        keep = cv2.dnn.NMSBoxes(
            bboxes=_boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.,
            nms_threshold=thresh,
            eta=1,
            top_k=5000
        )
        if len(keep) > 0:
            return keep.flatten()
        else:
            return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class YUNET():
    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        self.model_file = model_file
        self.nms_thresh = nms_thresh
        self.session = onnxruntime.InferenceSession(self.model_file, None)

        self.taskname = 'yunet'
        self.priors_cache = {}

    def anchor_fn(self, shape):
        min_sizes_cfg = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        steps = [8, 16, 32, 64]
        ratio = [1.]
        clip = False

        feature_map_2th = [int(int((shape[0] + 1) / 2) / 2), int(int((shape[1] + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2), int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2), int(feature_map_5th[1] / 2)]

        feature_maps = [feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = min_sizes_cfg[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    cx = (j + 0.5) * steps[k] / shape[1]
                    cy = (i + 0.5) * steps[k] / shape[0]
                    for r in ratio:
                        s_ky = min_size / shape[0]
                        s_kx = r * min_size / shape[1]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = np.array(anchors).reshape(-1, 4)
        if clip:
            output.clip(max=1, min=0)
        return output

    def forward(self, img, score_thresh, priors):
        img = np.transpose(img, [2, 0, 1]).astype(
            np.float32)[np.newaxis, ...].copy()

        loc, conf, iou = self.session.run(
            None, {self.session.get_inputs()[0].name: img})

        boxes = self.decode(loc, priors, variances=[0.1, 0.2])

        _, _, h, w = img.shape
        boxes[:, 0::2] *= w
        boxes[:, 1::2] *= h

        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]
        iou_scores = np.clip(iou_scores, a_min=0., a_max=1.)
        scores = np.sqrt(cls_scores * iou_scores)

        score_mask = scores > score_thresh
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        return boxes, scores

    def detect(self, img, score_thresh=0.5):
        if self.priors_cache.get(img.shape[:2], None) is None:
            priors = self.anchor_fn(img.shape[:2])
            self.priors_cache[img.shape[:2]] = priors
        else:
            priors = self.priors_cache[img.shape[:2]]

        bboxes, scores = self.forward(img, score_thresh, priors)

        pre_det = np.hstack((bboxes[:, :4], scores[:, None]))
        keep = nms(pre_det, self.nms_thresh)

        kpss = bboxes[keep, 4:]
        bboxes = pre_det[keep, :]

        return bboxes, kpss

    def decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        """
        boxes = loc.copy()
        boxes[:, 0:2] = priors[:, 0:2] + \
            boxes[:, 0:2] * variances[0] * priors[:, 2:4]
        boxes[:, 2:4] = priors[:, 2:4] * np.exp(boxes[:, 2:4] * variances[1])

        # (cx, cy, w, h) -> (x, y, w, h)
        boxes[:, 0:2] -= boxes[:, 2:4] / 2

        # xywh -> xyXY
        boxes[:, 2:4] += boxes[:, 0:2]
        # landmarks
        if loc.shape[-1] > 4:
            boxes[:, 4::2] = priors[:, None, 0] + \
                boxes[:, 4::2] * variances[0] * priors[:, None, 2]
            boxes[:, 5::2] = priors[:, None, 1] + \
                boxes[:, 5::2] * variances[0] * priors[:, None, 3]
        return boxes


class Detector:
    def __init__(self):
        self.d = YUNET('models/face_detection_yunet_2021dec.onnx', nms_thresh=0.3)

    def detect(self, image):
        # (l,t,r,b)
        faces, _ = self.d.detect(image, score_thresh=0.9)
        faces = faces[:, :4].astype(np.int32)
        # HACK negative coordinate
        faces = np.clip(faces, 0, None)
        # (l,t,r,b) -> (x,y,w,h)
        faces = np.array([[f[0], f[1], f[2]-f[0], f[3]-f[1]] for f in faces])

        return faces
