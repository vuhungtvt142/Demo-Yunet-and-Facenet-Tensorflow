import pickle
from time import time

import cv2 as cv
import numpy as np
from sklearn import neighbors

from detectors.dlib import hog_loc
from detectors.yunet import Detector
from encoders.facenet512 import Encoder
from utils.tracker import Tracker

detector = Detector()
tracker = Tracker()
encoder = Encoder()

print('Warming up...')

t = time()
for _ in range(10):
    detector.detect(np.zeros((1080//2, 1920//2, 3), dtype=np.uint8))

print(f'Warmup detector time: {time() - t:.2f} s')

t = time()
for _ in range(10):
    encoder.encode(np.zeros((160, 160, 3), dtype=np.uint8), [(0, 160, 160, 0)])

print(f'Warmup encoder time: {time() - t:.2f} s')


def distance(face_encodings, face_to_compare):
    face_encodings = [enc / np.linalg.norm(enc) for enc in face_encodings]
    face_to_compare = face_to_compare / np.linalg.norm(face_to_compare)
    return 1 - np.dot(face_encodings, face_to_compare)


seed_img = [
    {'class': '038', 'mean_dist': 0.09711340850288402},
    {'class': '047', 'mean_dist': 0.14754811724163291},
    {'class': '021', 'mean_dist': 0.14822813718619715},
    {'class': '002', 'mean_dist': 0.1556863428361144},
    {'class': '031', 'mean_dist': 0.1562856709592798},
    {'class': '037', 'mean_dist': 0.16895774619445153},
    {'class': '003', 'mean_dist': 0.17118144361605073},
    {'class': '040', 'mean_dist': 0.1766720240264818},
    {'class': '013', 'mean_dist': 0.17903064529110993},
    {'class': '017', 'mean_dist': 0.18001507998321528},
    {'class': '029', 'mean_dist': 0.1865894498409576},
    {'class': '039', 'mean_dist': 0.18964262178536773},
    {'class': '048', 'mean_dist': 0.1918764990597508},
    {'class': '014', 'mean_dist': 0.1961529882230436},
    {'class': '032', 'mean_dist': 0.1968796384079321},
    {'class': '018', 'mean_dist': 0.1972881095190054},
    {'class': '042', 'mean_dist': 0.2046173655354145},
    {'class': '016', 'mean_dist': 0.20604523695941335},
    {'class': '005', 'mean_dist': 0.2082604615226267},
    {'class': '015', 'mean_dist': 0.20845048197514796},
    {'class': '030', 'mean_dist': 0.21083674977970182},
    {'class': '027', 'mean_dist': 0.2135666744010798},
    {'class': '012', 'mean_dist': 0.2158322077634129},
    {'class': '022', 'mean_dist': 0.21838538658485704},
    {'class': '026', 'mean_dist': 0.2220392599809268},
    {'class': '000', 'mean_dist': 0.22244389591043767},
    {'class': '009', 'mean_dist': 0.22488934159577473},
    {'class': '043', 'mean_dist': 0.2337553948670169},
    {'class': '025', 'mean_dist': 0.234869251677757},
    {'class': '011', 'mean_dist': 0.23963302663958463},
    {'class': '033', 'mean_dist': 0.2411839818603439},
    {'class': '035', 'mean_dist': 0.24144250472923948},
    {'class': '001', 'mean_dist': 0.24748267485742764},
    {'class': '008', 'mean_dist': 0.2478044949222804},
    {'class': '024', 'mean_dist': 0.24800765284424373},
    {'class': '010', 'mean_dist': 0.25058811050010255},
    {'class': '046', 'mean_dist': 0.25541284269988285},
    {'class': '045', 'mean_dist': 0.2630546478107858},
    {'class': '006', 'mean_dist': 0.26402507071046993},
    {'class': '034', 'mean_dist': 0.26713474049849184},
    {'class': '023', 'mean_dist': 0.26733200882250246},
    {'class': '007', 'mean_dist': 0.2677468024501433},
    {'class': '019', 'mean_dist': 0.2684145321739642},
    {'class': '044', 'mean_dist': 0.2721576719069912},
    {'class': '050', 'mean_dist': 0.2822942953959966},
    {'class': '004', 'mean_dist': 0.285362405112256},
    {'class': '028', 'mean_dist': 0.28836709122370235},
    {'class': '041', 'mean_dist': 0.30068524399500957},
    {'class': '020', 'mean_dist': 0.30493546329886306},
    {'class': '036', 'mean_dist': 0.3472339783209199},
    {'class': '049', 'mean_dist': 0.3708875875288288},
]

seed_img_map = {
    s['class']: np.zeros((75, 75, 0), dtype=np.float32)
    for s in seed_img
}

seed_colors = {
    '006': 'red',  # Mr. Binh
    '038': 'red',  # Mr. Nhut
    '002': 'blue',  # Duy
    '004': 'blue',  # Linh
    '005': 'blue',  # Vy
    '008': 'blue',  # Mr. Thuong
    '011': 'blue',  # Ms. Diem
    '012': 'blue',  # Linh
    '029': 'blue',  # Ngoc
    '032': 'blue',  # An
    '034': 'blue',  # Ngan
    '035': 'blue',  # Ryan
    '037': 'blue',  # Dat
    '043': 'blue',  # Vinh
    '048': 'blue',  # Phong
}  # otherwise, on-seed: green, off-seed: yellow

t = time()
try:
    with open('seed.pickle', 'rb') as seed_file:
        seed_cls = pickle.load(seed_file)

    print('Load seed from cache!')
except:
    print('Generate seed!')
    seed_cls = neighbors.RadiusNeighborsClassifier(metric='cosine', radius=0.27, weights='distance')

print(f'Seed time: {time() - t:.2f} s')

face_id_count = 0
face_map = dict()  # face_id -> encode
face_img_map = dict()  # face_id -> img
face_last_img_map = dict()  # face_id -> img
recog_count = dict()  # face_id -> count
track_face_map = dict()  # track_id -> face_id
track_score_map = dict()  # track_id -> score
track_seed_map = dict()  # track_id -> seed_id
track_seed_label = dict()  # track_id -> seed_label

cam_tm = cv.TickMeter()
pro_tm = cv.TickMeter()

cap = cv.VideoCapture(0)

while cap.isOpened():
    cam_tm.stop()
    cam_tm.start()
    success, frame = cap.read()
    if not success:
        print('No frames grabbed!')
        break

    pro_tm.start()
    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = frame[:, :, ::-1]
    rgb_frame.flags.writeable = False

    faces = detector.detect(small_frame) * 2

    objects = tracker.update(faces)

    new_ids = objects.keys() - track_face_map.keys()

    def get_face_img_by_id(id):
        x, y, w, h = objects[id]
        t, l, b, r = (y, x, y+h, x+w)

        return frame[t:b, l:r]

    detectable_ids = [
        id for id in new_ids
        if len(hog_loc(get_face_img_by_id(id))) > 0
    ]

    face_locs = [objects[id] for id in detectable_ids]
    face_locs = [(y, x+w, y+h, x) for (x, y, w, h) in face_locs]

    face_encodings = encoder.encode(rgb_frame, face_locs)

    know_faces = list(face_map.values())
    face_ids = list(face_map.keys())

    # TODO improve performance MxN distance
    for tracking_id, face_encoding in zip(detectable_ids, face_encodings):
        if len(know_faces) == 0:
            face_map[face_id_count] = face_encoding
            face_img_map[face_id_count] = cv.resize(
                get_face_img_by_id(tracking_id),
                (75, 75),
            )
            track_face_map[tracking_id] = face_id_count
            track_score_map[tracking_id] = 0
            recog_count[face_id_count] = 1
            face_id_count += 1
            continue

        face_distances = distance(know_faces, face_encoding)
        best_match_index = np.argmin(face_distances)

        face_id = face_id_count
        if face_distances[best_match_index] < 0.32:
            face_id = face_ids[best_match_index]
            track_score_map[tracking_id] = face_distances[best_match_index]
            recog_count[face_id] += 1
        else:
            recog_count[face_id_count] = 1
            track_score_map[tracking_id] = 0
            face_id_count += 1

        face_map[face_id] = face_encoding
        if face_id in face_img_map:
            face_last_img_map[face_id] = face_img_map[face_id]
        face_img_map[face_id] = cv.resize(
            get_face_img_by_id(tracking_id),
            (75, 75),
        )
        track_face_map[tracking_id] = face_id

        try:
            cls = seed_cls.predict([face_encoding])[0]

            track_seed_map[tracking_id] = cls
            track_seed_label[tracking_id] = f'UID: {cls}'
        except ValueError:
            pass  # not-in-seed

    pro_tm.stop()

    preview = frame.copy()

    detected_faces = sorted([
        (track_face_map[id], id)
        for id in objects.keys()
        if id in track_face_map
    ])
    for no_face, (face_id, tracking_id) in enumerate(detected_faces):
        d, m = divmod(no_face, 12)
        pos_x = m * 150
        pox_y = d * 150
        preview[
            (40+pox_y):(115+pox_y), pos_x:(pos_x+75)
        ] = face_img_map[face_id]

        if face_id in face_last_img_map:
            preview[
                (40+pox_y):(115+pox_y), (pos_x+75):(pos_x+150)
            ] = face_last_img_map[face_id]
        '''
        if tracking_id in track_seed_label:
            preview[
                (115+pox_y):(190+pox_y), pos_x:(pos_x+75)
            ] = seed_img_map[track_seed_map[tracking_id]]
        '''
    for (id, (x, y, w, h)) in objects.items():
        t, l, b, r = (y, x, y+h, x+w)
        cx, cy = (x + w//2, y + h//2)

        cv.rectangle(preview, (l, t), (r, b), (128, 128, 128), 1)

        label = f'SID: {track_face_map[id]}/{recog_count[track_face_map[id]]}/{track_score_map[id]:.2f}' if id in track_face_map else 'Tracking...'
        cv.putText(
            preview, label,
            (cx - 10, cy),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)
        )

        label2 = track_seed_label[id] if id in track_seed_label else 'No contact'
        cv.putText(
            preview, label2,
            (cx - 10, cy + 15),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)
        )

        cv.circle(preview, (cx, cy), 2, (0, 255, 0), -1)

    for (x, y, w, h) in faces:
        cv.rectangle(preview, (x, y), (x+w, y+h), (255, 255, 255), 1)

    for (id, (x, y, w, h)) in objects.items():
        if id not in track_face_map:
            continue

        bgr = (0, 255, 255)

        if id in track_seed_map:
            seed_id = track_seed_map[id]

            t, l, b, r = (y, x, y+h, x+w)
            cx, cy = (x + w//2, y + h//2)

            bgr = (0, 255, 0)
            if seed_id in seed_colors:
                color = seed_colors[seed_id]
                if color == 'red':
                    bgr = (0, 0, 255)
                elif color == 'blue':
                    bgr = (255, 0, 0)

        cv.rectangle(preview, (l, t), (r, b), bgr, 1)

    cv.putText(
        preview,
        f'Overall FPS: {cam_tm.getFPS():.2f}',
        (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
    )
    cv.putText(
        preview,
        f'Process FPS: {pro_tm.getFPS():.2f}',
        (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
    )

    # Visualize results in a new Window
    cv.imshow('ActiveCam', preview)

    if cv.waitKey(5) & 0xFF == 27:
        break

    # pro_tm.reset()

cv.destroyAllWindows()

cap.release()
