import unittest
import geone
import numpy as np

import os
TEST_DIR = os.path.dirname(__file__)

def read(filename, dtype=None):
    """
    Function that reads a gslib to a dictionary of numpy arrays
    """
    with open(filename, 'r') as file_handle:
        number_of_variables, column_names = _read_header_info(file_handle)
        output = np.loadtxt(file_handle, dtype=dtype)
    return { variable: values for variable, values in zip(column_names, output.T)}

def _read_header_info(file_handle):
    file_handle.readline()
    number_of_variables = int(file_handle.readline().strip().split()[0])
    return number_of_variables, [file_handle.readline().strip().split()[0] for line_number in range(number_of_variables)]

def write(dictionary, filename):
    """
    Writes dictionary of numpy arrays to a gslib
    """
    with open(filename, 'w') as gslib_file:
        _write_header_info(dictionary, gslib_file)
        np.savetxt(gslib_file, np.transpose([dictionary[key] for key in dictionary]))
    return

def _write_header_info(dictionary, file_handle):
    file_handle.write("Standard gslib created by geone package\n")
    file_handle.write(str(len(dictionary))+'\n')
    for column_name in dictionary:
        file_handle.write(str(column_name)+'\n')


class TestReadWrite(unittest.TestCase):
    def setUp(self):
        self.gslib_content = {
                'X': [13.5, 61.5, 76.5,],
                'Y': [82.5, 60.5, 1.5,],
                'Z': [0.5, 0.5, 0.5,],
                'code_real00000': [0, 1, 1,],
                }
        self.temp_dir = os.path.join(TEST_DIR, 'temp')

    def test_read(self):
        gslib = read(os.path.join(TEST_DIR, 'data/sample_test.gslib'))
        np.testing.assert_equal(self.gslib_content, gslib)

    def test_write(self):
        # Create a temporary directory for gslib out
        os.makedirs(self.temp_dir)

        # Write the output file 
        self.output_file = os.path.join(self.temp_dir, 'test_output.gslib')
        write(self.gslib_content, filename=self.output_file)

        # Read it back and compare the reference
        gslib = read(self.output_file)
        np.testing.assert_equal(self.gslib_content, gslib)

    def tearDown(self):
        try:
            os.remove(self.output_file)
        except (AttributeError, FileNotFoundError):
            pass

        try:
            os.rmdir(self.temp_dir)
        except FileNotFoundError:
            pass


class TestPointSetToDict(unittest.TestCase):
    def setUp(self):
        self.gslib_content = {
                'X': [13.5, 61.5, 76.5,],
                'Y': [82.5, 60.5, 1.5,],
                'Z': [0.5, 0.5, 0.5,],
                'code_real00000': [0, 1, 1,],
                }

    def test_compare_with_read(self):
        point_set = geone.img.readPointSetGslib(os.path.join(TEST_DIR, 'data/sample_test.gslib'))
        np.testing.assert_equal(self.gslib_content, point_set.to_dict())
