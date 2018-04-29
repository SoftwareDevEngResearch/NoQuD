#!/usr/bin/env python

import numpy as np
from numba import jitclass          # import the decorator
from numba import int64, float32

spec = [
    ('repeated_coefficients', float32[:, :]),               # a simple scalar field
    ('diffusion_constant', float32[:, :]),
    ('sigma_r', float32[:, :]),
    ('cell_size', float32[:, :]),
    ('f', float32[:, :]),
    ('groups', int64),
    ('nodes', int64),
    ('order_of_legendre_poly', int64),
    ('linear_system', float32[:, :, :]),

]

@jitclass(spec)
class Nodal(object):

    def __init__(self, diffusion_constant, sigma_r, cell_size, f, groups, nodes):

        self.repeated_coefficients = np.array(([0, 1, -3, 6, -10],
                                 [0, 1, 3, 6, 10],
                                 [0, 0, -12, 0, -40],
                                 [0, 0, 0, -60, 0],
                                 [0, 0, 0, 0, -140],
                                 [0, 1, -3, 6, -10]), dtype=np.float32)

        self.diffusion_constant = diffusion_constant
        self.sigma_r = sigma_r
        self.cell_size = cell_size
        self.f = f
        self.groups = groups
        self.nodes = nodes
        self.order_of_legendre_poly = 5
        self.linear_system = np.zeros((self.groups, self.order_of_legendre_poly * self.nodes,
                                       self.order_of_legendre_poly * self.nodes), dtype=np.float32)
        self.build_linear_system()



    @property
    def lhs_boundary_condition(self):

        for i in xrange(0, self.groups):
            for j in xrange(0, self.order_of_legendre_poly):

                self.linear_system[i][0][j] = self.repeated_coefficients[0][j]

    def rhs_boundary_condition(self):

        for i in xrange(0, self.groups):
            for j in xrange(0, self.order_of_legendre_poly):

                self.linear_system[i][1][-j] = self.repeated_coefficients[1][-j]

    def flux_interface_condition(self):

        for i in xrange(0, self.groups):
            for j in xrange(0, self.nodes - 1):
                for k in xrange(0, self.order_of_legendre_poly):

                    rhs = self.repeated_coefficients[1][k]*self.diffusion_constant[i][j]/self.cell_size[i][j]
                    self.linear_system[i][2 + j][self.order_of_legendre_poly * j + k] = rhs

                    rhs = self.repeated_coefficients[5][k]*self.diffusion_constant[i][j + 1]/self.cell_size[i][j + 1]
                    self.linear_system[i][2 + j][self.order_of_legendre_poly * (j + 1) + k] = rhs

    def current_interface_condition(self):

        for i in xrange(0, self.groups):
            for j in xrange(0, self.nodes - 1):
                for k in xrange(0, self.order_of_legendre_poly):

                    rhs1 = f[i][2 * j]
                    self.linear_system[i][self.nodes + 1 + j][self.order_of_legendre_poly * j + k] = rhs1

                    rhs2 = pow(-1, k) * f[i][2 * j + 1]
                    self.linear_system[i][self.nodes + 1 + j][self.order_of_legendre_poly * (j + 1) + k] = rhs2

    def build_linear_system(self):

        self.lhs_boundary_condition
        self.rhs_boundary_condition()
        self.flux_interface_condition()
        self.current_interface_condition()

if __name__ == "__main__":

    diffusion_coefficient = np.array([[1, 1], [1, 1]], dtype=np.float32)
    sigma_r = np.array([[1, 1], [1, 1]], dtype=np.float32)
    cell_size = np.array([[1, 1], [1, 1]], dtype=np.float32)
    f = np.array([[1, 1], [1, 1]], dtype=np.float32)
    groups = int(2)
    nodes = int(2)
    test = Nodal(diffusion_coefficient, sigma_r, cell_size, f, groups, nodes)
    # test.lhs_boundary_condition()
    # test.rhs_boundary_condition()
    # test.flux_interface_condition()
    # test.current_interface_condition()
    #test.build_linear_system()
    print test.linear_system
