from NoQuD.read_input_data import read_csv_input_file as read_csv
from NoQuD.StepCharacteristicSolve.StepCharacteristicSolver import *
import os
import csv


def test_integration():

    """" This tests to see if data extracted from the input file can be used to create a StepCharacteristicSolver
    object without error. """

    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'AI_test.csv')

    sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, assembly_map, material, assembly_size, \
    assembly_cells = read_csv.read_csv(file_path)

    slab = StepCharacteristicSolver(sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, material)
    slab.solve()

def test_step_characteristic_solve():

    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'AI_test.csv')

    sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, assembly_map, material, assembly_size, \
    assembly_cells = read_csv.read_csv(file_path)

    slab = StepCharacteristicSolver(sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, material)
    slab.solve()

    rows = []  # initialize a temporary storage variable
    elements = []  # initialize a temporary storage variable

    file_path = os.path.join(current_directory, 'validation_flux.csv')

    # Read in flux validation data.
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            rows.append(row[:1])
            for col in row:
                elements.append(col)

    number_rows = len(rows)
    number_elements = len(elements)

    # Reshape matrix to correct format.
    flux_ref = numpy.reshape(elements, [number_rows, number_elements / number_rows]).astype(numpy.float)

    file_path = os.path.join(current_directory, 'validation_k.csv')

    # Read in eigenvalue validation data.
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            k_ref = row

    k_ref = float(k_ref[0])  # convert to float

    flux_difference = abs(flux_ref - slab.flux_new)  # calculate difference in flux
    k_difference = abs(k_ref - slab.k_new) # calculate difference in eigenvalue

    # If the fluxes are within 1e-5 and the eigenvalues are within 1e-4, we'll say it works.

    assert numpy.max(flux_difference) < 1e-5
    assert k_difference < 1e-4


def test_more_than_two_assemblies():

    """" This tests to see if data extracted from the input file with more than two assemblies can be used to create a
    StepCharacteristicSolver object without error. """

    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'AI_3plus_test.csv')

    sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, assembly_map, material, assembly_size, \
    assembly_cells = read_csv.read_csv(file_path)

    slab = StepCharacteristicSolver(sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, material)
    slab.solve()

    # If one wants to inspect the data, uncomment the lines below.
#     x = numpy.arange(0.0, 10.0*len(assembly_map), 10.0*len(assembly_map) / cells)
#     plt.plot(x, slab.flux_new[0][:])
#     plt.plot(x, slab.flux_new[1][:])
#     plt.xlabel('Position [cm]')
#     plt.ylabel('Flux [s^-1 cm^-2]')
#     plt.title('Neutron Flux')
#     plt.show()
#     print "Multiplication Factor: {0}".format(slab.k_new)
#
# if __name__ =="__main__":
#     test_integration()
#     test_step_characteristic_solve()
#     test_more_than_two_assemblies()
