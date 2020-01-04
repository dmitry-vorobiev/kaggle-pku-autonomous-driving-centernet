import numpy as np


def parse_annot_str(s):
    cars = np.array(s.split()).reshape([-1, 7])
    out = []
    for car in cars:
        car = {
            'car_id': int(car[0]),
            'rotation': car[1:4].astype(np.float64),
            'location': car[4:7].astype(np.float64),
        }
        out.append(car)
    return out


def pad_img_sides(img, pad_pct):
    """ Pad image on left and right
        Input:
            img - image of shape (C, H, W)
            pad_pct - portion of added width relative to the original width
        Output:
            img
    """
    c, h, w = img.shape
    empty_shape = (c, h, int(w * pad_pct / 2))
    empty = np.ones(empty_shape, dtype=img.dtype) * img.mean(2, keepdims=True)
    img = np.concatenate([empty, img, empty], axis=2)
    return img
