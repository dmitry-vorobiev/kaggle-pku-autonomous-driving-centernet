import numpy as np

def parse_annot_str(s):
    cars = np.array(s.split()).reshape([-1,7])
    out = []
    for car in cars:
        car = {'car_id': int(car[0]), 'pose': car[1:].astype(np.float64)}
        out.append(car)
    return out