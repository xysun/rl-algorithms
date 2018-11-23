'''
Class constructors:
    - a 2D continuous space: ([lower_x, lower_y], [upper_x, upper_y])
    - grid_size n: each tile will cover 1/n of bounded distance in each dimension
    - tile offsets [(x,y)]
methods:
    - `encode((x,y))`: return a (n*d) dimension numpy vector with one hot encoding for each tile,
        where `n` is number of tiles, d is from one hot encoding

Boundaries are detected with [), i.e. closed on left, open on right
'''

import numpy as np
from sklearn.preprocessing import OneHotEncoder


class TileEncoder(object):
    def __init__(self, lower_x, lower_y, upper_x, upper_y, n, tile_offsets:[(int, int)]):

        self.encoders = [SingleTileEncoder(lower_x, lower_y, upper_x, upper_y, n)]

        self.step_x = (upper_x - lower_x) / n
        self.step_y = (upper_y - lower_y) / n

        for (offset_x, offset_y) in tile_offsets:
            self.encoders.append(
                SingleTileEncoder(
                    lower_x=lower_x + offset_x * self.step_x,
                    lower_y=lower_y + offset_y * self.step_y,
                    upper_x=upper_x + offset_x * self.step_x,
                    upper_y=upper_y + offset_y * self.step_y,
                    n=n
                )
            )

        self.d = n * 2 * (1+len(tile_offsets))

    def encode(self, x, y):

        return np.array([ohc.encode(x, y) for ohc in self.encoders])


class SingleTileEncoder(object):
    def __init__(self, lower_x, lower_y, upper_x, upper_y, n):
        # can't do tuple unpacking in Python like `__init__(self, (lower_x, lower_y))` :(
        self.lower_x = lower_x
        self.lower_y = lower_y
        self.upper_x = upper_x
        self.upper_y = upper_y
        self.n: int = n

        self.step_x = (self.upper_x - self.lower_x) / self.n
        self.step_y = (self.upper_y - self.lower_y) / self.n

        # fit one hot encoder
        self.boundaries_x = self.get_boundaries(self.lower_x, self.upper_x, self.n)
        self.boundaries_y = self.get_boundaries(self.lower_y, self.upper_y, self.n)

        boundaries = []
        for x in self.boundaries_x:
            for y in self.boundaries_y:
                boundaries.append(['x-' + x, 'y-' + y])
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(boundaries)

    @staticmethod
    def get_boundaries(lower, upper, n):
        step = (upper - lower) / n
        result = []
        for i in range(0, n):
            result.append("{0:.2f}".format(lower + i*step))
        return result

    def encode(self, x, y):

        if x < self.lower_x or x > self.upper_x:
            x_in = 'x-unknown'
        else:
            x_in = 'x-' + self.boundaries_x[int((x - self.lower_x) / self.step_x)]

        if y < self.lower_y or y > self.upper_y:
            y_in = 'y-unknown'
        else:
            y_in = 'y-' + self.boundaries_y[int((y - self.lower_y) / self.step_y)]

        return self.ohe.transform([[x_in, y_in]]).toarray()[0]
