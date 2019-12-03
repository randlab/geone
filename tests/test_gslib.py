import unittest
import geone
import numpy as np

import os
TEST_DIR = os.path.dirname(__file__)

class TestGslibReadWrite(unittest.TestCase):
    def setUp(self):
        self.gslib_content = {
                'X': [13.5, 61.5, 76.5,],
                'Y': [82.5, 60.5, 1.5,],
                'Z': [0.5, 0.5, 0.5,],
                'code_real00000': [0, 1, 1,],
                }
        self.temp_dir = os.path.join(TEST_DIR, 'temp')

    def test_read(self):
        gslib = geone.gslib.read(os.path.join(TEST_DIR, 'data/sample_test.gslib'))
        np.testing.assert_equal(self.gslib_content, gslib)

    def test_write(self):
        # Create a temporary directory for gslib out
        os.makedirs(self.temp_dir)

        # Write the output file 
        self.output_file = os.path.join(self.temp_dir, 'test_output.gslib')
        geone.gslib.write(self.gslib_content, filename=self.output_file)

        # Read it back and compare the reference
        gslib = geone.gslib.read(self.output_file)
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
