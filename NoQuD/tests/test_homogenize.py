#!/usr/bin/env python

import NoQuD.homogenize.homogenize as homo
import os

# Still need tests for the homogenized results.


def test_homogenize_assembly_running():

    """ Make sure the HomogenizeAssembly class can be initialized (which also performs the homogenization) without
        error. """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    local_path = os.path.join('testing_files', 'assembly_info_single_test.csv')
    file_path = os.path.join(current_dir, local_path)
    test = homo.HomogenizeAssembly(file_path)


def test_homogenize_global_running():

    """ Make sure the HomogenizeGlobal class can be initialized (which also performs the homogenization) without
        error. """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    local_path1 = os.path.join('testing_files', 'assembly_info_test.csv')
    local_path2 = os.path.join('testing_files', 'assembly_info_single_test.csv')
    file_path1 = os.path.join(current_dir, local_path1)
    file_path2 = os.path.join(current_dir, local_path2)
    test = homo.HomogenizeGlobe([file_path1, file_path2, file_path2])


if __name__=="__main__":
    test_homogenize_assembly_running()
    test_homogenize_global_running()