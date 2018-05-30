#!/usr/bin/env python

import NoQuD.nodal.Nodal as Nodal
import numpy as np


def test_nodal_running():

    """ Make sure the Nodal class can be initialized without error."""

    diffusion_coefficient = np.array([[1, 1], [1, 1]], dtype=np.float64)
    sigma_r = np.array([[1, 1], [1, 1]], dtype=np.float64)
    cell_size = np.array([[1, 1], [1, 1]], dtype=np.float64)
    f = np.array([[1, 1], [1, 1]], dtype=np.float64)
    groups = int(2)
    nodes = int(2)
    test = Nodal.Nodal(diffusion_coefficient, sigma_r, cell_size, f, groups, nodes)


