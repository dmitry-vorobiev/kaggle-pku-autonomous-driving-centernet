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
