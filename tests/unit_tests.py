import unittest
import random

import numpy as np
from ppca import PPCA


class NaNTest(unittest.TestCase):

    def remove_nan_test(self):

        N = 101
        k = 23
        p_nan = 0.02
        n_components = 3

        data = np.random.random((N, k))
        for n in range(N):
            for _k in range(k):
                if random.random() < p_nan:
                    data[n, _k] = np.nan

        pca = PPCA()
        pca.fit(data, n_components)

        self.assertEqual(pca.data[np.isnan(pca.data)].shape, (0, ))
        self.assertEqual(pca.C.shape, (k, n_components))
        self.assertEqual(pca.transform().shape, (N, n_components))
