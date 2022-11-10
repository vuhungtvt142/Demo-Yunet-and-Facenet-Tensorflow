import dlib

hog_detector = dlib.get_frontal_face_detector()


def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def hog_loc(img):
    return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in hog_detector(img, 1)]
