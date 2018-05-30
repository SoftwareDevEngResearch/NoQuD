import NoQuD.HomogeneousSolve.NodalSolve as NS
import os


def test_nodal_solve_running():

    """ Make sure the NodalSolve class can be initialized without error."""

    current_dir = os.path.dirname(os.path.realpath(__file__))
    local_path1 = os.path.join('testing_files', 'assembly_info_test.csv')
    local_path2 = os.path.join('testing_files', 'assembly_info_single_test.csv')
    file_path1 = os.path.join(current_dir, local_path1)
    file_path2 = os.path.join(current_dir, local_path2)

    test = NS.NodalSolve([file_path1, file_path2, file_path2])
    test.solve()
