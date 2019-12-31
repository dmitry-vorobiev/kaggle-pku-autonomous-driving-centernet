from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data


class KaggleCars(data.Dataset):
    def __init__(self, cfg):
        super(KaggleCars, self).__init__()

    def __len__(self):
        pass