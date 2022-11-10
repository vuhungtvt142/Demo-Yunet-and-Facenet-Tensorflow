from scipy.spatial import distance as dist
import numpy as np


class Tracker():
    def __init__(self, max_disappeared=10):
        self.count = 0
        self.objects = dict()
        self.object_rects = dict()
        self.disappeared = dict()
        self.disappeared2 = set()

        self.max_disappeared = max_disappeared

    def register(self, centroid, rect):
        self.objects[self.count] = centroid
        self.object_rects[self.count] = rect
        self.disappeared[self.count] = 0
        self.count += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.object_rects[object_id]
        del self.disappeared[object_id]

        self.disappeared2.add(object_id)

    def _disapeared(self):
        result = list(self.disappeared2)
        self.disappeared2 = set()

        return result

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.object_rects

        input_centroids = np.column_stack((
            rects[:, 0] + rects[:, 2] // 2,
            rects[:, 1] + rects[:, 3] // 2,
        ))

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i])

            return self.object_rects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        D = dist.cdist(object_centroids, input_centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            x, y, w, h = rects[col]
            new_ox, new_oy = x + w//2, y + h//2
            ox, oy = object_centroids[row]

            if ox < new_ox - w or ox > new_ox + w or oy < new_oy - h or oy > new_oy + h:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.object_rects[object_id] = rects[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1

            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        for col in unused_cols:
            self.register(input_centroids[col], rects[col])

        return self.object_rects
