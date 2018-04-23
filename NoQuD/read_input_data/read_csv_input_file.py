#!/usr/bin/env python

# Import pandas
import pandas as pd
import numpy as np


def read_csv(filename):

    # Load csv
    try:
        data = pd.read_csv(filename, header = None)
    except IOError:
        raise IOError("Make sure that: \n (1) the input file is in the local directory. \n (2) the input file is in "
                      "CSV format.")

    # Assign singular values from csv.
    try:
        assembly_types = int(data.iloc[0, 1])  # number of assembly types.
        assemblies = int(data.iloc[1, 1])  # number of total assemblies
        assembly_size = int(data.iloc[2, 1])  # size of assemblies
        cells = int(data.iloc[3, 1])  # number of cells
        groups = int(data.iloc[4, 1])  # number of neutron energy groups
        unique_materials = int(data.iloc[5, 1])  # number of unique materials
        assembly_cells = int(cells / assemblies)  # the number of cells per assembly.
        cell_size = assemblies * assembly_size / cells  # length of each cell
    except IndexError:
        raise IndexError("The input file may have incorrect formatting. Check these fields: \n"
                         "Number of Assembly Types, Number of Assemblies, Assembly Size, Number of Total Cells, Number"
                         "of Unique Materials.")
    except ValueError:
        raise ValueError("The input file may have incorrect data types. Check these fields: \n Number of Assembly "
                         "Types, Number of Assemblies, Assembly Size, Number of Total Cells, Number of Unique Materials"
                         ".")


    # Assign values that requiring looping.
    sig_t, sig_sin, sig_sout, sig_f, nu, chi = assign_cross_sections(data, groups, unique_materials)
    key_length = assign_key_length(data, assembly_types)
    material, assembly_map = create_material_map(data, assemblies, assembly_types, assembly_cells, key_length, cells)

    print "File loaded."

    # Return relevant parameters.
    return sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, assembly_map.astype(int), material, \
           assembly_size, assembly_cells


def assign_cross_sections(data, groups, unique_materials):

    # Initialize matrices to hold nuclear data in each unique material in each energy group
    sig_t = np.zeros([groups, unique_materials])  # total macro cross section (mcs)
    sig_sin = np.zeros([groups, unique_materials])  # in-scatter mcs
    sig_sout = np.zeros([groups, unique_materials])  # out-scatter mcs
    sig_f = np.zeros([groups, unique_materials])  # fission mcs
    nu = np.zeros([groups, unique_materials])  # avg. number of neutrons generated per fission
    chi = np.zeros([groups, unique_materials])  # probability for a fission neutron to appear in an energy group

    try:
        for i in xrange(0, groups):  # loop over groups
            for j in xrange(0, unique_materials):  # loop over unique materials
                sig_t[i][j] = data.iloc[9, 1 + i + 2 * j]
                sig_sin[i][j] = data.iloc[10, 1 + i + 2* j]
                sig_sout[i][j] = data.iloc[11, 1 + i + 2 * j]
                sig_f[i][j] = data.iloc[12, 1 + i + 2 * j]
                nu[i][j] = data.iloc[13, 1 + i + 2 * j]
                chi[i][j] = data.iloc[14, 1 + i + 2 * j]
    except IndexError:
        raise IndexError("The input file may have incorrect formatting. Make sure the nuclear data is correctly "
                         "formatted.")
    except ValueError:
        raise ValueError("The input file may have incorrect data types. Make sure the nuclear data is correctly "
                         "formatted.")


    return sig_t, sig_sin, sig_sout, sig_f, nu, chi


def create_material_map(data, assemblies, assembly_types, assembly_cells, key_length, cells):

    local_map = np.zeros([assembly_types, assembly_cells])  # material map for each assembly
    assembly_map = np.zeros([assemblies])  # coarse map for order of assemblies

    try:
        # Now we build a material map for each assembly.
        for i in xrange(0, assembly_types):  # loop over assembly types
            for j in xrange(0, int(assembly_cells / key_length[i])):  # loop over number of key lengths in each assembly
                #  type
                for k in xrange(0, int(key_length[i])):  # loop over key lengths
                    local_map[i][j * int(key_length[i]) + k] = data.iloc[21 + 4 * i, k + 1]

        for i in xrange(0, assemblies):
            assembly_map[i] = data.iloc[17, i + 1]

        # The local assembly maps are then used to make a global material map from the geometry described in the
        # Assembly Map entry of the csv.
        material = np.array(local_map[int(data.iloc[17, 1]) - 1][:])  # initialize as first assembly in geometry

        # This loops concatenates additional assemblies to the global map as specified in the Assembly Map entry.
        for i in xrange(1, assemblies):
            material = np.concatenate((material, local_map[int(data.iloc[17, i + 1]) - 1][:]))
    except IndexError:
        raise IndexError("The input file may have incorrect formatting. Make sure the key and assembly map are "
                         "correctly formatted.")
    except ValueError:
        raise ValueError("The input file may have incorrect data types. Make sure the key and assembly map are "
                         "correctly formatted.")

    # Subtract one from each material value to reflect the correct index.
    material = material - np.ones([cells])

    return material.astype(int), assembly_map.astype(int)


def assign_key_length(data, assembly_types):
    key_length = np.zeros([assembly_types])  # the key length describes the length of periodicity in the material map
    try:
        for i in xrange(0, assembly_types):
            key_length[i] = int(data.iloc[20 + 4 * i, 1])
    except IndexError:
        raise IndexError("The input file may have incorrect formatting. Make sure the key lengths are correctly "
                         "formatted.")
    except ValueError:
        raise ValueError("The input file may have incorrect data types. Make sure the key lengths are correctly "
                         "formatted.")

    return key_length


if __name__ == "__main__":
    print read_csv("AI_test.csv")