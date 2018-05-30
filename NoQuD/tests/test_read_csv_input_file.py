#!/usr/bin/env python
from NoQuD.read_input_data.read_csv_input_file import assign_key_length, assign_cross_sections, create_material_map
import pandas as pd
import numpy as np
import os


def test_assign_key_length():

    """" Tests the key_length function from read_csv_input_file.py """

    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'testing_files/assembly_info_test.csv')

    data = pd.read_csv(file_path, header = None)
    assembly_types = int(data.iloc[0, 1])
    obv_key_length = assign_key_length(data, assembly_types)
    exp_key_length = np.array([8.0, 8.0])  # Values to expect from examining the input file.

    for i in xrange(0, len(obv_key_length)):
        assert obv_key_length[i] == exp_key_length[i]

def test_create_material_map():

    """" Tests the create_material_length function from read_csv_input_file.py """

    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'testing_files/assembly_info_test.csv')

    data = pd.read_csv(file_path, header=None)
    assemblies = int(data.iloc[1, 1])
    assembly_types = int(data.iloc[0, 1])
    cells = int(data.iloc[3, 1])
    assembly_cells = int(cells / assemblies)
    key_length = assign_key_length(data, assembly_types)
    obs_material, obs_assembly_map = create_material_map(data, assemblies, assembly_types, assembly_cells, key_length,
                                                         cells)
    exp_material = [2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0,
        2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0,
        0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2,
        1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2,
        2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
        2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2]  # Values to expect from examining the input file.
    exp_assembly_map = [1, 2]

    for i in xrange(0, len(obs_material)):
        assert obs_material[i] == exp_material[i]

    for i in xrange(0, len(obs_assembly_map)):
        assert obs_assembly_map[i] == exp_assembly_map[i]

def test_assign_cross_sections():

    """" Tests the assign_cross_sections function from read_csv_input_file.py """

    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'testing_files/assembly_info_test.csv')

    data = pd.read_csv(file_path, header=None)
    groups = int(data.iloc[4, 1])  # number of neutron energy groups
    unique_materials = int(data.iloc[5, 1])  # number of unique materials
    obs_sig_t, obs_sig_sin, obs_sig_sout, obs_sig_f, obs_nu, obs_chi = assign_cross_sections(data, groups,
                                                                                             unique_materials)
    # Values to expect from examining the input file.
    exp_sig_t = np.array([[0.2, 0.2, 0.2], [0.6, 0.2, 1.1]])
    exp_sig_sin = np.array([[0.2, 0.2, .17], [0, 0, 1.1]])
    exp_sig_sout = np.array([[0, 0, 0.03], [0, 0, 0]])
    exp_sig_f = np.array([[0, 0, 0], [0.6, 0.2, 0]])
    exp_nu = np.array([[0, 0, 0], [1.5, 1.5, 0]])
    exp_chi = np.array([[1, 1, 0], [0, 0, 0]])

    for i in xrange(0, unique_materials):
        for j in xrange(0, groups):
            assert exp_sig_t[j][i] == obs_sig_t[j][i]
            assert exp_sig_sin[j][i] == obs_sig_sin[j][i]
            assert exp_sig_sout[j][i] == obs_sig_sout[j][i]
            assert exp_sig_f[j][i] == obs_sig_f[j][i]
            assert exp_nu[j][i] == obs_nu[j][i]
            assert exp_chi[j][i] == obs_chi[j][i]


if __name__ == '__main__':

     test_assign_key_length()
     test_create_material_map()
     test_assign_cross_sections()
