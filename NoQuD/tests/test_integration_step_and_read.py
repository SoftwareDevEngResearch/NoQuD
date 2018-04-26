from NoQuD.read_input_data import read_csv_input_file as read_csv
from NoQuD.StepCharacteristicSolve.StepCharacteristicSolver import *
import os


def test_integration():

    """" This tests to see if data extracted from the input file can be used to create a StepCharacteristicSolver
    object without error. """

    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'AI_test.csv')

    print file_path

    sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, assembly_map, material, assembly_size, \
    assembly_cells = read_csv.read_csv(file_path)

    slab = StepCharacteristicSolver(sig_t, sig_sin, sig_sout, sig_f, nu, chi)
