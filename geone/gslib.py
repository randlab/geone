import numpy as np

def read(filename, dtype=None):
    """
    Function that reads a gslib to a dictionary of numpy arrays
    """
    with open(filename, 'r') as file_handle:
        number_of_variables, column_names = read_header_info_(file_handle)
        output = np.loadtxt(file_handle, dtype=dtype)
    return { variable: values for variable, values in zip(column_names, output.T)}

def read_header_info_(file_handle):
    file_handle.readline()
    number_of_variables = int(file_handle.readline().strip().split()[0])
    return number_of_variables, [file_handle.readline().strip().split()[0] for line_number in range(number_of_variables)]

def write(dictionary, filename):
    with open(filename, 'w') as gslib_file:
        write_header_info_(dictionary, gslib_file)
        np.savetxt(gslib_file, np.transpose([dictionary[key] for key in dictionary]))
    return


def write_header_info_(dictionary, file_handle):
    file_handle.write("Standard gslib created by geone package\n")
    file_handle.write(str(len(dictionary))+'\n')
    for column_name in dictionary:
        file_handle.write(str(column_name)+'\n')
