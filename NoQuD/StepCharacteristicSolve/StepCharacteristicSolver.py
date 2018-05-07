#!/usr/bin/env python

#Consider vectorizing with Numpy
#Also consider using pypy

import numpy
import matplotlib.pyplot as plt
from numba import jitclass, int64, float64

spec = [
    ('sig_t', float64[:, :]),
    ('sig_s_in', float64[:, :]),
    ('sig_s_out', float64[:, :]),
    ('sig_f', float64[:, :]),
    ('nu', float64[:, :]),
    ('chi', float64[:, :]),
    ('ab', float64[:]),
    ('weights', float64[:]),
    ('groups', int64),
    ('core_mesh_length', int64),
    ('dx', float64),
    ('dmu', float64),
    ('flux_new', float64[:, :]),
    ('flux_old', float64[:, :]),
    ('phi_L_old', float64[:, :]),
    ('phi_R_old', float64[:, :]),
    ('angular_flux_edge', float64[:, :, :]),
    ('angular_flux_center', float64[:, :, :]),
    ('k_old', float64),
    ('k_new', float64),
    ('spatial_fission_old', float64[:, :]),
    ('spatial_fission_new', float64[:, :]),
    ('material', int64[:]),
    ('exit1', int64),
    ('exit2', int64),
    ('flux_iterations', int64),
    ('source_iterations', int64),
    ('Q', float64[:, :]),
    ('fission_source_dx', float64),
    ('spatial_sig_s_out', float64[:, :])

]


@jitclass(spec)
class StepCharacteristicSolver(object):

    # Initialize and assign variables.
    def __init__(self, sig_t, sig_s_in, sig_s_out, sig_f, nu, chi, groups, cells, cell_size, material):

        # Nuclear data
        self.sig_t = sig_t  # total cross section
        self.sig_s_in = sig_s_in  # within group scatter cross section
        self.sig_s_out = sig_s_out  # out of group scatter cross section
        self.sig_f = sig_f  # fission cross section
        self.nu = nu  # number of neutrons produced per fission
        self.chi = chi  # probability of fission neutrons appearing in each group

        # Quadrature data
        self.ab = numpy.array([-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472,
                               -0.1488743389816312, 0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
                               0.8650633666889845, 0.9739065285171717], dtype=numpy.float64)
        self.weights = numpy.array([0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963,
                                    0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
                                    0.1494513491505806, 0.0666713443086881], dtype=numpy.float64)

        # Problem geometry parameters
        self.groups = groups  # energy groups in problem
        self.core_mesh_length = cells  # number of intervals
        self.dx = cell_size  # discretization in length
        self.dmu = 2 / len(self.ab) # discretization in angle

        # Set initial values
        self.flux_new = numpy.ones((self.groups, self.core_mesh_length), dtype=numpy.float64)  # initialize flux
        self.flux_old = numpy.ones((self.groups, self.core_mesh_length), dtype=numpy.float64)  # initialize flux
        self.phi_L_old = numpy.ones((self.groups, len(self.ab) / 2),
                                    dtype=numpy.float64)  # initialize left boundary condition
        self.phi_R_old = numpy.ones((self.groups, len(self.ab) / 2),
                                    dtype=numpy.float64)  # initialize right boundary condition
        self.angular_flux_edge = numpy.zeros((self.groups, self.core_mesh_length + 1, len(self.ab)),
                                             dtype=numpy.float64)  # initialize edge flux
        self.angular_flux_center = numpy.zeros((self.groups, self.core_mesh_length, len(self.ab)),
                                               dtype=numpy.float64)  # initialize edge flux
        self.k_old = 1.0  # initialize eigenvalue
        self.k_new = 1.0  # initialize eigenvalue
        self.spatial_sig_s_out = numpy.zeros((self.groups, self.core_mesh_length),dtype=numpy.float64)
        self.spatial_fission_old = numpy.zeros((self.groups, self.core_mesh_length),
                                               dtype=numpy.float64)  # initialize fission source
        self.spatial_fission_new = numpy.zeros((self.groups, self.core_mesh_length),
                                               dtype=numpy.float64)  # initialize fission source
        self.material = material  # material map
        # self.material = numpy.array([int(2)] * int(self.core_mesh_length), dtype=numpy.int64)  # material map
        # self.E = numpy.zeros([self.groups, self.core_mesh_length])  # initialize eddington factors
        self.fission_source_dx = 0.0

        # Solver metrics
        self.exit1 = 0  # initialize exit condition
        self.exit2 = 0  # initialize exit condition
        self.flux_iterations = 0  # iteration counter
        self.source_iterations = 0  # iteration counter

        # Form spatial matrix for in scatter from other groups.
        self.form_scatter_source()

        # Form fission sources for each group.
        self.form_fission_source()
        self.spatial_fission_old = self.spatial_fission_new

        # Form combined source
        self.Q = self.spatial_sig_s_out + self.spatial_fission_old / self.k_old

        # Implement initial angular fluxes conditions at cell edges.
        for k in xrange(self.groups):
            for j in [0, self.core_mesh_length]:
                for i in xrange(10):
                    if i + 1 <= len(self.ab) / 2 and j == self.core_mesh_length:
                        self.angular_flux_edge[k][j][i] = self.phi_R_old[k][i]
                    elif i + 1 > len(self.ab) / 2 and j == 0:
                        self.angular_flux_edge[k][j][i] = self.phi_L_old[k][i - 5]

    # With a given flux, the source from scattering is calculated.
    def form_scatter_source(self):
        # Form scattering source.
        self.spatial_sig_s_out = numpy.zeros((self.groups, self.core_mesh_length), dtype=numpy.float64)
        for i in xrange(self.core_mesh_length):
            for k in xrange(self.groups):
                self.spatial_sig_s_out[1 - k][i] = self.dx * self.flux_old[k][i] * self.sig_s_out[k][self.material[i]]

    # With a given flux, the source from fission is calculated.
    def form_fission_source(self):
        # Form fission source.
        self.spatial_fission_new = numpy.zeros((self.groups, self.core_mesh_length), dtype=numpy.float64)
        for i in xrange(self.core_mesh_length):
            for k in xrange(self.groups):
                for h in xrange(self.groups):
                    self.fission_source_dx = self.dx * self.flux_old[h][i] * self.chi[k][self.material[i]] * self.nu[h][
                        self.material[i]] * self.sig_f[h][
                                                 self.material[i]]
                    self.spatial_fission_new[k][i] = self.spatial_fission_new[k][i] + self.fission_source_dx

    # Generate a standard pin assembly material map of a 2 assembly problem.
    # Note: not used anymore
    def material_map(self):

        for i in xrange(8):
            for j in xrange(4):
                cell = 2 + 8 * i + j
                self.material[cell] = 0  # 0 represents fuel 1.

        for i in xrange(8):
            for j in xrange(4):
                cell = 66 + 8 * i + j
                self.material[cell] = 1  # 1 represents fuel 2.

        # 2 represents water, which occupies all other spaces.

    # With given angular fluxes, calculate the scalar flux using a quadrature set.
    def calculate_scalar_flux(self):

        for k in xrange(self.groups):
            for i in xrange(self.core_mesh_length):
                for x in xrange(len(self.ab)):
                    self.flux_new[k][i] = self.flux_new[k][i] + self.weights[x] * self.angular_flux_center[k][i][x]

    # Reflects given angular fluxes at the boundary across the mu = 0 axis.
    def assign_boundary_condition(self):
        # Redefine angular flux at boundaries.
        for k in xrange(self.groups):
            for j in xrange(0, self.core_mesh_length+1):
                for i in xrange(10):
                    if i + 1 <= len(self.ab) / 2 and j == self.core_mesh_length:
                        self.angular_flux_edge[k][j][i] = self.angular_flux_edge[k][j][len(self.ab) - i - 1]
                    elif i + 1 > len(self.ab) / 2 and j == 0:
                        self.angular_flux_edge[k][j][i] = self.angular_flux_edge[k][j][len(self.ab) - i - 1]

    # Propagate angular flux boundary conditions across the problem.
    def flux_iteration(self):
        for k in xrange(self.groups):
            for z in xrange(5, 10):
                for i in xrange(self.core_mesh_length):
                    self.angular_flux_edge[k][i + 1][z] = self.angular_flux_edge[k][i][z] * numpy.exp(
                        -self.sig_t[k][self.material[i]] * self.dx / abs(self.ab[z])) + ((
                            self.dx * self.sig_s_in[k][self.material[i]] * self.flux_old[k][i] + self.Q[k][i])) / (
                                                                  2 * self.dx * self.sig_t[k][self.material[i]]) * (
                                                                  1 - numpy.exp(
                                                              - self.sig_t[k][self.material[i]] * self.dx / abs(
                                                                  self.ab[z])))

                    n = 0.5 * self.dx * self.sig_s_in[k][self.material[i]] * self.flux_old[k][i] + 0.5 * self.Q[k][
                        i] - self.ab[z] * (
                                self.angular_flux_edge[k][i + 1][z] - self.angular_flux_edge[k][i][z])

                    d = self.sig_t[k][self.material[i]] * self.dx

                    self.angular_flux_center[k][i][z] = n / d

            for z in xrange(0, 5):
                for i in range(self.core_mesh_length, 0, -1):
                    self.angular_flux_edge[k][i - 1][z] = self.angular_flux_edge[k][i][z] * numpy.exp(
                        -self.sig_t[k][self.material[i - 1]] * self.dx / abs(self.ab[z])) + (
                                                                  (self.dx * self.sig_s_in[k][
                                                                      self.material[i - 1]] * self.flux_old[k][
                                                                       i - 1] + self.Q[k][i - 1]) / (
                                                                          2 * self.dx * self.sig_t[k][
                                                                      self.material[i - 1]])) * (1 - numpy.exp(
                        -self.sig_t[k][self.material[i - 1]] * self.dx / abs(self.ab[z])))

                    n = 0.5 * self.dx * self.sig_s_in[k][self.material[i - 1]] * self.flux_old[k][i - 1] + 0.5 * \
                        self.Q[k][i - 1] - \
                        self.ab[z] * (self.angular_flux_edge[k][i][z] - self.angular_flux_edge[k][i - 1][z])

                    d = self.sig_t[k][self.material[i - 1]] * self.dx

                    self.angular_flux_center[k][i - 1][z] = n / d

        self.calculate_scalar_flux()

    # With a given scalar flux, calculate a eigenvalue and source with a power iteration.
    def source_iteration(self):

        # New eigenvalue.
        self.k_new = self.k_old * numpy.sum(self.spatial_fission_new[0][:]) / numpy.sum(self.spatial_fission_old[0][:])

        # New source.
        self.Q = (self.spatial_sig_s_out + self.spatial_fission_new / self.k_new)

    # Using all the methods above, solve for an eigenvalue and flux with defined convergence criteria.
    def solve(self):
        print "Sit tight. This takes a while."

        while self.exit2 == 0:  # source convergence

            self.source_iterations += 1

            while self.exit1 == 0:  # flux convergence

                self.flux_iterations += 1
                self.flux_iteration()  # do a flux iteration

                # Check for convergence
                if abs(numpy.max(((self.flux_new[0][:] - self.flux_old[0][:]) / self.flux_new[0][:]))) < 1E-6 and abs(
                        numpy.max(((self.flux_new[1][:] - self.flux_old[1][:]) / self.flux_new[1][:]))) < 1E-6:
                    self.exit1 = 1  # exit flux iteration
                    self.flux_old = self.flux_new # assign flux

                else:
                    self.flux_old = self.flux_new  # assign flux
                    self.flux_new = numpy.zeros((self.groups, self.core_mesh_length), dtype=numpy.float64)  # reset new_flux
                    self.assign_boundary_condition()

            # Form scattering source.
            self.form_scatter_source()

            # Form fission source.
            self.form_fission_source()

            # Do a power iteration.
            self.source_iteration()

            # Check for convergence of new eigen value and fission source
            if abs(self.k_new - self.k_old) / self.k_old < 1.0E-5 and numpy.max(
                    self.spatial_fission_old[0][:] - self.spatial_fission_new[0][:]) < 1.0E-5:

                self.exit2 = 1  # exit source iteration
                self.flux_new = self.flux_new / (numpy.sum(self.flux_new)) # normalize flux

            else:

                # Reassign parameters to iterate again.
                self.k_old = self.k_new
                self.spatial_sig_s_out = numpy.zeros((self.groups, self.core_mesh_length), dtype=numpy.float64)
                self.flux_new = numpy.ones((self.groups, self.core_mesh_length), dtype=numpy.float64)
                self.spatial_fission_old = self.spatial_fission_new
                self.spatial_fission_new = numpy.zeros((self.groups, self.core_mesh_length), dtype=numpy.float64)
                self.exit1 = 0  # reenter flux iteration loop.

    # Plot and display results.
    # Note: doesn't work with numba
    def results(self):

        print 'Eigenvalue: {0}'.format(self.k_new)
        print 'Source iterations: {0}'.format(self.source_iterations)
        print 'Computation time: {:04.2f} seconds'.format(self.end - self.start)

        x = numpy.arange(0.0, 20., 20.0 / 128.0)
        plt.plot(x, self.flux_new[0][:])
        plt.plot(x, self.flux_new[1][:])
        plt.xlabel('Position [cm]')
        plt.ylabel('Flux [s^-1 cm^-2]')
        plt.title('Neutron Flux')
        plt.show()








