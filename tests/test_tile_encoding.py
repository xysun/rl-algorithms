import unittest

import numpy as np

from common.tile_encoding import SingleTileEncoder, TileEncoder


class TestTileEncoderModule(unittest.TestCase):

    def test_get_boundaries(self):
        result = SingleTileEncoder.get_boundaries(0, 1.3, 3)
        self.assertEqual(result, ["0.00", "0.43", "0.87"])

    def test_float_boundary(self):
        encoder = SingleTileEncoder(
            lower_x=0,
            lower_y=0,
            upper_x=1.3,
            upper_y=1.1,
            n=3
        )

        self.assertTrue(np.array_equal(encoder.encode(0.45, 0.8), np.array([0, 1, 0, 0, 0, 1])))

    def test_single_tile_encoding(self):
        encoder = SingleTileEncoder(
            lower_x=2,
            lower_y=2,
            upper_x=4,
            upper_y=4,
            n=2
        )

        self.assertTrue(np.array_equal(encoder.encode(0, 0), np.array([0, 0, 0, 0])))  # out-of-bound

        self.assertTrue(np.array_equal(encoder.encode(2, 2), np.array([1, 0, 1, 0])))
        self.assertTrue(np.array_equal(encoder.encode(2, 3), np.array([1, 0, 0, 1])))
        self.assertTrue(np.array_equal(encoder.encode(3, 2), np.array([0, 1, 1, 0])))
        self.assertTrue(np.array_equal(encoder.encode(3, 3), np.array([0, 1, 0, 1])))

        self.assertTrue(np.array_equal(encoder.encode(3.2, 3.2), np.array([0, 1, 0, 1])))

    def test_uniform_offset(self):
        encoder = TileEncoder(
            lower_x=0,
            lower_y=0,
            upper_x=2,
            upper_y=2,
            n=2,
            tile_offsets=[(1, 1)]
        )

        # [0,0,2,2], [1,1,3,3]
        result1 = encoder.encode(0, 0)
        self.assertEqual(result1.shape, (2, 4))
        self.assertTrue(np.array_equal(result1, np.array([
            [1, 0, 1, 0],
            [0, 0, 0, 0]
        ])))

        result2 = encoder.encode(1.2, 1.2)
        self.assertTrue(np.array_equal(result2, np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])))

        # self.assertTrue(np.array_equal(encoder.encode(0, 0), np.array([0] * 4)))

    def test_asymmetric_offset(self):
        encoder = TileEncoder(
            lower_x=0,
            lower_y=0,
            upper_x=3,
            upper_y=2,
            n=3,
            tile_offsets=[(2, 1)]
        )

        # [0,0,3,2], [2,2,4,3]
        result = encoder.encode(1, 1)
        self.assertTrue(np.array_equal(result, np.array([
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0]
        ])))

    def test_multiple_offsets(self):
        encoder = TileEncoder(
            lower_x=0,
            lower_y=0,
            upper_x=2,
            upper_y=2,
            n=2,
            tile_offsets=[(1, 1), (2, 2)]
        )

        result = encoder.encode(0, 0)
        self.assertEqual(result.shape, (3, 4))
