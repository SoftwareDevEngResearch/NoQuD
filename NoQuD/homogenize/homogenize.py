#!/usr/bin/env python

import numpy as np
from NoQuD.read_input_data import read_csv_input_file as read_csv
from NoQuD.step_characteristic_solve.StepCharacteristicSolver import *
import os

""" Contains classes for homogenizing 1D problems. """

class HomogenizeAssembly:

    """ Solves the heterogeneous problem, and uses the solution to calculate homogeneous nuclear data.

    Args:
        assembly_data_file (str): filename for the assembly to be homogenized.

    Attributes:
        sig_t (float[][]): total macroscopic nuclear cross section in each material in each energy group
        sig_sin (float[][]): in group scatter macroscopic nuclear cross section in each material in each energy group
        sig_sout (float[][]): out of group scatter macroscopic nuclear cross section in each material in each energy group
        sig_f (float[][]): fission macroscopic nuclear cross section in each material in each energy group
        nu (float[][]): average number of neutron born per fission in each material in each energy group
        chi (float[][]): probability for a fission neutron to be born in each material in each energy group
        groups (int): number of energy groups
        cells (int): total number of cells
        cell_size (float): size of each cell
        assembly_map (int[]): describes the layout of assemblies in the problem. Each unique integer corresponds to a
            unique assembly type.
            EX: If assembly_map = [1, 2, 1], this corresponds to problem with 2 unique assemblies. The assembly of type
            2 is sandwiched between two assemblies of type 1.
        material (int[]): describes the layout of materials in the problem. Each unique integer corresponds to a unique
            material. Physical interpretation is the same as for assembly_map.
        assembly_size (float): the size of each assembly
        assembly_cells (int): the number of cells in each assembly
        slab (StepCharacteristicSolver instance): contains the heterogeneous to the problem
        average_edge_flux (float): the average flux calculated using all edge flux values
        discontinuity_factor_left ([float[]]): discontinuity factors on left for the flux in each energy group
        discontinuity_factor_right ([float[]]): discontinuity factors on right for the flux in each energy group
        sig_t_h (float[]): homogeneous (flux-weighted) total macroscopic nuclear cross section for entire assembly in
            each energy group.
        sig_sin_h (float[]): homogeneous (flux-weighted) in group scatter macroscopic nuclear cross section for entire
            assembly in each energy group
        sig_sout_h (float[]): (float[]): homogeneous (flux-weighted) out of group macroscopic nuclear cross section for
            entire assembly in each energy group
        sig_f_h (float[]): homogeneous (flux-weighted) fission macroscopic nuclear cross section for entire assembly in
            each energy group
        eddington_factor_h (float[]): homogeneous (flux-weighted) eddington factors for the entire assembly in each
            energy group
        sig_r_h (float[]): homogeneous (flux-weighted) removal macroscopic cross section for entire assembly in each
            energy group
    """

    def __init__(self, assembly_data_file):

        # Determine current directory path for input file and read in data, then use that data to create a
        # StepCharacteristicSolver instance, where the heterogeneous solution is found.
        file_path = assembly_data_file
        self.sig_t, self.sig_sin, self.sig_sout, self.sig_f, self.nu, self.chi, self.groups, self.cells, self.cell_size\
            , self.assembly_map, self.material, self.assembly_size, self.assembly_cells = read_csv.read_csv(file_path)
        self.slab = StepCharacteristicSolver(self.sig_t, self.sig_sin, self.sig_sout, self.sig_f, self.nu, self.chi,
                                             self.groups, self.cells, self.cell_size, self.material)
        self.slab.solve()

        # Initialize variables
        self.average_edge_flux = np.zeros((2, 1))
        self.discontinuity_factor_left = np.zeros((2, 1))
        self.discontinuity_factor_right = np.zeros((2, 1))
        self.sig_t_h = np.zeros((2, 1))
        self.sig_sin_h = np.zeros((2, 1))
        self.sig_sout_h = np.zeros((2, 1))
        self.sig_f_h = np.zeros((2, 1))
        self.eddington_factor_h = np.zeros((2, 1))
        self.sig_r_h = np.zeros((2, 1))

        # Calculate the homogeneous data.
        self.perform_homogenization()

    def calculate_homogenized_nuclear_data(self):

        """
        Calculates the homogenized nuclear data using flux weighting. (This preserves reaction rates.)
        """

        for group in xrange(self.groups):
            for cell in xrange(self.cells):
                self.sig_t_h[group][0] = self.sig_t_h[group][0] + self.sig_t[group][self.material[cell]]\
                                      * self.slab.flux_new[group][cell]/np.sum(self.slab.flux_new[group][:])
                self.sig_sin_h[group][0] = self.sig_sin_h[group][0] + self.sig_sin[group][self.material[cell]] \
                                        * self.slab.flux_new[group][cell] / np.sum(self.slab.flux_new[group][:])
                self.sig_sout_h[group][0] = self.sig_sout_h[group][0] + self.sig_sout[group][self.material[cell]] \
                                      * self.slab.flux_new[group][cell] / np.sum(self.slab.flux_new[group][:])
                self.sig_f_h[group][0] = self.sig_f_h[group][0] + self.sig_f[group][self.material[cell]] \
                                      * self.slab.flux_new[group][cell] / np.sum(self.slab.flux_new[group][:])
                self.eddington_factor_h[group][0] = self.eddington_factor_h[group][0] + \
                                                 self.slab.eddington_factors[group][cell] * \
                                                 self.slab.flux_new[group][cell] / np.sum(self.slab.flux_new[group][:])
                self.sig_r_h = self.sig_t_h - self.sig_sin_h

    def calculate_discontinuity_factors(self):

        """
        Calculates the discontinuity factors on each side of the assembly by dividing the edge flux by the average flux.
        """

        for group in xrange(self.groups):
            self.average_edge_flux[group] = np.mean(self.slab.edge_flux[group, :])
            self.discontinuity_factor_left[group] = self.slab.edge_flux[group][self.cells]/self.average_edge_flux[group]
            self.discontinuity_factor_right[group] = self.slab.edge_flux[group][0]/self.average_edge_flux[group]

    def perform_homogenization(self):

        """
        Runs all methods required to homogenize data.
        """

        self.calculate_homogenized_nuclear_data()
        self.calculate_discontinuity_factors()


class HomogenizeGlobe:

    """ Creates multiple HomogenizeAssembly instances and stitches the data to create a homogeneous problem description.

    Args:
        single_assembly_input_files (str[]): contains the global assembly data file, followed by the assembly
            information files for each unique assembly.

    Attributes:

    Note: the attributes below are for the global assembly data

        sig_t (float[][]): total macroscopic nuclear cross section in each material in each energy group
        sig_sin (float[][]): in group scatter macroscopic nuclear cross section in each material in each energy group
        sig_sout (float[][]): out of group scatter macroscopic nuclear cross section in each material in each energy group
        sig_f (float[][]): fission macroscopic nuclear cross section in each material in each energy group
        nu (float[][]): average number of neutron born per fission in each material in each energy group
        chi (float[][]): probability for a fission neutron to be born in each material in each energy group
        groups (int): number of energy groups
        cells (int): total number of cells
        cell_size (float): size of each cell
        assembly_map (int[]): describes the layout of assemblies in the problem. Each unique integer corresponds to a
            unique assembly type.
            EX: If assembly_map = [1, 2, 1], this corresponds to problem with 2 unique assemblies. The assembly of type
            2 is sandwiched between two assemblies of type 1.
        material (int[]): describes the layout of materials in the problem. Each unique integer corresponds to a unique
            material. Physical interpretation is the same as for assembly_map.
        assembly_size (float): the size of each assembly
        assembly_cells (int): the number of cells in each assembly

    Note: The attributes below will hold the homogenized nuclear data for each unique assembly. 'a' stands for assembly.

        sig_t_a (float[]): homogeneous (flux-weighted) total macroscopic nuclear cross section for each unique
            assembly in each energy group.
        sig_sin_a (float[]): homogeneous (flux-weighted) in group scatter macroscopic nuclear cross section for a single
            assembly in each energy group
        sig_sout_a (float[]): (float[]): homogeneous (flux-weighted) out of group macroscopic nuclear cross section for
            each unique assembly in each energy group
        sig_f_a (float[]): homogeneous (flux-weighted) fission macroscopic nuclear cross section for each unique assembly in
            each energy group
        eddington_factor_a (float[]): homogeneous (flux-weighted) eddington factors for the each unique assembly in each
            energy group
        sig_r_a (float[]): homogeneous (flux-weighted) removal macroscopic cross section for each unique assembly in each
            energy group
        f_a (float[][]): left ([:][0]) and right discontinuity factors ([:][1]) for each unique assembly.

    Note: The attributes below hold the homogenized nuclear data for each homogenized node in the problem. 'g' stands
        for global

        sig_t_g (float[]): homogeneous (flux-weighted) total macroscopic nuclear cross section for each node in
            each energy group.
        sig_sin_g (float[]): homogeneous (flux-weighted) in group scatter macroscopic nuclear cross section for each
            unique assembly in each energy group
        sig_sout_g (float[]): (float[]): homogeneous (flux-weighted) out of group macroscopic nuclear cross section for
            each node in each energy group
        sig_f_g (float[]): homogeneous (flux-weighted) fission macroscopic nuclear cross section for each node in
            each energy group
        eddington_factor_g (float[]): homogeneous (flux-weighted) eddington factors for the each node in each
            energy group
        sig_r_g (float[]): homogeneous (flux-weighted) removal macroscopic cross section for each node in each
            energy group
        f_g (float[][]): left ([:][0]) and right discontinuity factors ([:][1]) for each node.

    """
    def __init__(self, single_assembly_input_files):

        # Determine local directory to find input file.
        self.single_assembly_input_files = single_assembly_input_files
        file_path = single_assembly_input_files[0]
        self.sig_t, self.sig_sin, self.sig_sout, self.sig_f, self.nu, self.chi, self.groups, self.cells, self.cell_size \
            , self.assembly_map, self.material, self.assembly_size, self.assembly_cells = read_csv.read_csv(file_path)

        # 'a' stands for assembly. The variables below will hold the homogenized nuclear data for each unique assembly.
        self.eddington_factor_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_r_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_t_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_sin_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_sout_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_f_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.f_a = np.zeros((self.groups, 2*(len(single_assembly_input_files)-1)))

        # 'g' stand for global. The variables below will hold the homogenized nuclear data for each node.
        self.eddington_factor_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_r_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_t_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_sin_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_sout_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_f_g = np.zeros((self.groups, len(self.assembly_map)))
        self.f_g = np.zeros((self.groups, 2*len(self.assembly_map)))

        # Run methods to perform global homogenization.
        self.build_assembly_data()
        self.build_global_data()

    def build_assembly_data(self):

        """ Calculates and stores homogenized data for each unique assembly.
        Creates a HomogenizeAssembly instance for each unique assembly and stores the data into HomogenizeGlobe
        attributes.
        """

        for index in xrange(len(self.single_assembly_input_files)-1):
            assembly = HomogenizeAssembly(self.single_assembly_input_files[index+1])
            self.eddington_factor_a[:, index-1] = assembly.eddington_factor_h[:, 0]
            self.sig_r_a[:, index] = assembly.sig_r_h[:, 0]
            self.sig_t_a[:, index] = assembly.sig_t_h[:, 0]
            self.sig_sin_a[:, index] = assembly.sig_sin_h[:, 0]
            self.sig_sout_a[:, index] = assembly.sig_sout_h[:, 0]
            self.sig_f_a[:, index] = assembly.sig_f_h[:, 0]
            self.f_a[:, index*2] = assembly.discontinuity_factor_left[:, 0]
            self.f_a[:, index*2 + 1] = assembly.discontinuity_factor_right[:, 0]

    def build_global_data(self):

        """
        Stitches homogenized data of unique assemblies together in order dictated by assembly_map.
        """

        for node in xrange(len(self.assembly_map)):
            self.eddington_factor_g[:, node] = self.eddington_factor_a[:, self.assembly_map[node] - 1]
            self.sig_r_g[:, node] = self.sig_r_a[:, self.assembly_map[node] - 1]
            self.sig_t_g[:, node] = self.sig_t_a[:, self.assembly_map[node] - 1]
            self.sig_sin_g[:, node] = self.sig_sin_a[:, self.assembly_map[node] - 1]
            self.sig_sout_g[:, node] = self.sig_sout_a[:, self.assembly_map[node] - 1]
            self.sig_f_g[:, node] = self.sig_f_a[:, self.assembly_map[node] - 1]
            self.f_g[:, 2*node:2*node + 2] = self.f_a[:, 2*(self.assembly_map[node] - 1):2*self.assembly_map[node] - 1]
