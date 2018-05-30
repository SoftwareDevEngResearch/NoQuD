#!/usr/bin/env python

""" Builds a coefficient matrix for nodal methods."""

import numpy as np
from numba import jitclass          # import the decorator
from numba import int64, float64

# spec = [
#     ('repeated_coefficients', float64[:, :]),               # a simple scalar field
#     ('diffusion_constant', float64[:, :]),
#     ('sigma_r', float64[:, :]),
#     ('cell_size', float64[:, :]),
#     ('f', float64[:, :]),
#     ('groups', int64),
#     ('nodes', int64),
#     ('order_of_legendre_poly', int64),
#     ('linear_system', float64[:, :, :]),
# ]

#@jitclass(spec)
class Nodal(object):

    """ Builds the coefficient matrix of the linear system required to perform a nodal diffusion solve.

    Args:
        diffusion_constant (float[][]): homogenized diffusion constant or eddington factor for each assembly in each
            energy group.
        sigma_r (float[][]): homogenized macroscopic removal cross section for each assembly in each energy group.
        node_size (float[][]): the size of each node in each energy group. (size should be the same between groups for
            the same assembly.)
        f (float[][]): discontinuity factors for each assembly.
        groups (int): number of energy groups
        nodes (int): number of nodes in system

    Attributes:
        repeated_coefficients (float[][]): coefficients repeated in the elements of the coefficient matrix.
        diffusion_constant (float[][]): homogenized diffusion constant or eddington factor for each assembly in each
            energy group.
        sigma_r (float[][]): homogenized macroscopic removal cross section for each assembly in each energy group.
        node_size (float[][]): the size of each node in each energy group. (size should be the same between groups for
            the same assembly.)
        f (float[][]): discontinuity factors for each assembly.
        groups (int): number of energy groups
        nodes (int): number of nodes in system
        order_of_legendre_poly (int): order of legendre polynomial used to approximate the homogenized flux. Only tested
            for order_of_legendre_poly = 5.
        linear_system [float[][][]]: contains the coefficient matrix used in the nodal solve for the fast and slow
            energy groups.

    """

    def __init__(self, diffusion_constant, sigma_r, node_size, f, groups, nodes):

        # Explicitly define repeated coefficients.
        self.repeated_coefficients = np.array(([0, 1, -3, 6, -10],
                                 [0, 1, 3, 6, 10],
                                 [0, 0, -12, 0, -40],
                                 [0, 0, 0, -60, 0],
                                 [0, 0, 0, 0, -140],
                                 [0, -1, 3, -6, 10]), dtype=np.float64)

        # Assign attributes from instance arguments.
        self.diffusion_constant = diffusion_constant
        self.sigma_r = sigma_r
        self.node_size = node_size
        self.f = f
        self.groups = groups
        self.nodes = nodes
        self.order_of_legendre_poly = 5

        # Initialize coefficient matrix.
        self.linear_system = np.zeros((self.groups, self.order_of_legendre_poly * self.nodes,
                                       self.order_of_legendre_poly * self.nodes), dtype=np.float64)

        # Build the linear system.
        self.build_linear_system()

    def lhs_boundary_condition(self):

        """ Asserts a reflecting boundary condition on the left boundary. """

        for i in xrange(0, self.groups):
            for j in xrange(0, self.order_of_legendre_poly):
                self.linear_system[i][0][j] = self.repeated_coefficients[0][j]

    def rhs_boundary_condition(self):

        """ Asserts a reflecting boundary condition on the right boundary. """

        for i in xrange(0, self.groups):
            for j in xrange(0, self.order_of_legendre_poly):
                self.linear_system[i][1][-j] = self.repeated_coefficients[1][-j]

    def flux_interface_condition(self):

        """ Asserts continuity of flux at interior node boundaries """

        for i in xrange(0, self.groups):
            for j in xrange(0, self.nodes - 1):
                for k in xrange(0, self.order_of_legendre_poly):

                    rhs = self.repeated_coefficients[1][k]*self.diffusion_constant[i][j]/self.node_size[i][j]
                    self.linear_system[i][2 + j][self.order_of_legendre_poly * j + k] = rhs

                    rhs = self.repeated_coefficients[5][k]*self.diffusion_constant[i][j + 1]/self.node_size[i][j + 1]
                    self.linear_system[i][2 + j][self.order_of_legendre_poly * (j + 1) + k] = rhs

    def current_interface_condition(self):

        """ Asserts discontinuous current at interior node boundaries."""

        for i in xrange(0, self.groups):
            for j in xrange(0, self.nodes - 1):
                for k in xrange(0, self.order_of_legendre_poly):

                    rhs1 = self.f[i][2 * j]
                    self.linear_system[i][self.nodes + 1 + j][self.order_of_legendre_poly * j + k] = rhs1

                    rhs2 = pow(-1, k+1) * self.f[i][2 * j + 1]
                    self.linear_system[i][self.nodes + 1 + j][self.order_of_legendre_poly * (j + 1) + k] = rhs2

    def balance_condition(self):

        """ Asserts conservation of neutrons in each node. """

        for i in xrange(0, self.groups):
            for j in xrange(0, self.nodes):
                for k in xrange(0, self.order_of_legendre_poly):

                    self.linear_system[i][2 * self.nodes + j][5 * j + k] = self.repeated_coefficients[2][k] * \
                                                                           self.diffusion_constant[i][j] / \
                                                                           self.node_size[i][j] ** 2
                self.linear_system[i][2 * self.nodes + j][5 * j] = self.sigma_r[i][j]

    def first_weighted_moment(self):

        """
        Asserts conservation of the first weighted moment of the neutron quasi-diffusion/diffusion equation
        in each node.
        """

        for i in xrange(0, self.groups):
            for j in xrange(0, self.nodes):
                for k in xrange(0, self.order_of_legendre_poly):

                    self.linear_system[i][3 * self.nodes + j][5 * j + k] = self.repeated_coefficients[3][k] * \
                                                                           self.diffusion_constant[i][j] / \
                                                                           self.node_size[i][j] ** 2
                self.linear_system[i][3 * self.nodes + j][5 * j + 1] = self.sigma_r[i][j]

    def second_weighted_moment(self):

        """ Asserts conservation of the second weighted moment of the neutron quasi-diffusion equation in each node"""

        for i in xrange(0, self.groups):
            for j in xrange(0, self.nodes):
                for k in xrange(0, self.order_of_legendre_poly):

                    self.linear_system[i][4 * self.nodes + j][5 * j + k] = self.repeated_coefficients[4][k] * \
                                                                           self.diffusion_constant[i][j] / \
                                                                           self.node_size[i][j] ** 2
                self.linear_system[i][4 * self.nodes + j][5 * j + 2] = self.sigma_r[i][j]

    def build_linear_system(self):

        """ Runs methods necessary to build coefficient matrix in its entirety. """

        self.lhs_boundary_condition()
        self.rhs_boundary_condition()
        self.flux_interface_condition()
        self.current_interface_condition()
        self.balance_condition()
        self.first_weighted_moment()
        self.second_weighted_moment()


if __name__ == "__main__":

    diffusion_coefficient = np.array([[1, 1], [1, 1]], dtype=np.float64)
    sigma_r = np.array([[1, 1], [1, 1]], dtype=np.float64)
    cell_size = np.array([[1, 1], [1, 1]], dtype=np.float64)
    f = np.array([[1, 1], [1, 1]], dtype=np.float64)
    groups = int(2)
    nodes = int(2)
    test = Nodal(diffusion_coefficient, sigma_r, cell_size, f, groups, nodes)
    # test.lhs_boundary_condition()
    # test.rhs_boundary_condition()
    # test.flux_interface_condition()
    # test.current_interface_condition()
    #test.build_linear_system()
    print test.linear_system
