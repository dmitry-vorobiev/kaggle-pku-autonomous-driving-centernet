import json
import numpy as np
import os
from collections import OrderedDict
from utils import car_models


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


def parse_pred_str(s):
    cars = np.array(s.split()).reshape([-1, 7])
    out = []
    for car in cars:
        car = {
            'rotation': car[0:3].astype(np.float64),
            'location': car[3:6].astype(np.float64),
            'score': car[-1].astype(np.float64),
        }
        out.append(car)
    return out


def load_car_models(model_dir):
    """Load all the car models"""
    car_models_all = OrderedDict([])
    for model in car_models.models:
        car_model = os.path.join(model_dir, model.name+'.json')
        with open(car_model) as json_file:
            car = json.load(json_file)
        for key in ['vertices', 'faces']:
            car[key] = np.array(car[key])
        # fix the inconsistency between obj and pkl
        car['vertices'][:, [0, 1]] *= -1
        car_models_all[model.name] = car 
    return car_models_all
