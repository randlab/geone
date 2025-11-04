# For printing messages, use:
#    pip install . --verbose
# or
#    pip install . -v

import sys, os, platform
import setuptools

COL_CONFIG = '\033[1;36m\033[40m' # bold cyan on black
COL_ERR = '\033[1;91m'            # high-intensity bold red
COL_RESET = '\033[0m'

linux_glibc_version = (2, 35)     # glibc version for the library in */linux_py*/
linux_old_glibc_version = (2, 27) # glibc version for the library in */linux_old_py*/

# Get configuration: platform system, version of python, version of glibc
try:
    platform_system = platform.system()     # 'Windows' or 'Linux' or 'Darwin' accepted
    python_version = sys.version_info[0:2]  # e.g. (3, 10)

    try:
        glibc_str = os.confstr('CS_GNU_LIBC_VERSION')                     # string like 'glibc 2.35'
        glibc_version = tuple([int(s) for s in glibc_str[6:].split('.')]) # e.g. (2, 35)
    except:
        glibc_version = None

    try:
        machine = platform.uname().machine
    except:
        machine = None

except:
    sys.exit(f'{COL_ERR}ERROR: getting config{COL_RESET}')

print(f'{COL_CONFIG}Your configuration: platform_system={platform_system}, python_version={python_version}, glibc_version={glibc_version}{COL_RESET}')

# Set subdir_selected: name of the subdirectory containing the C libraries to be installed,
# according to the platform system, the version of python and the version of glibc
# [subdir_selected = <prefix>_<suffix>]

# Set prefix
if platform_system == 'Windows':
    prefix = 'win'
elif platform_system == 'Linux':
    if glibc_version is not None:
        if glibc_version[0] > linux_glibc_version[0] or (glibc_version[0] == linux_glibc_version[0] and glibc_version[1] >= linux_glibc_version[1]):
            prefix = 'linux'
        elif glibc_version[0] > linux_old_glibc_version[0] or (glibc_version[0] == linux_old_glibc_version[0] and glibc_version[1] >= linux_old_glibc_version[1]):
            prefix = 'linux_old'
        else:
            prefix = None
    else:
        prefix = 'linux' # default
elif platform_system == 'Darwin':
    if machine == 'x86_64':
        prefix = 'mac_x86_64'
    elif machine == 'arm64':
        prefix = 'mac_arm64'
    else:
        prefix = None
else:
    prefix = None

# Set suffix
suffix = f'py{python_version[0]}{python_version[1]}'

# Set subdir_selected
if prefix is None or suffix is None:
    subdir_selected = None
else:
    subdir_selected = f'{prefix}_{suffix}'

# Set directories containing the libraries of the right version
deesse_core_dir_selected = f'src/geone/lib_deesse_core/{subdir_selected}'
geosclassic_core_dir_selected = f'src/geone/lib_geosclassic_core/{subdir_selected}'

if subdir_selected is None or not os.path.isdir(deesse_core_dir_selected) or not os.path.isdir(geosclassic_core_dir_selected):
    sys.exit(f'{COL_ERR}ERROR: package geone not available for your configuration [platform_system={platform_system}, python_version={python_version}, glibc_version={glibc_version}]{COL_RESET}')

# Set long_description
with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

# Load version
__version__ = '0.0.0' # default
with open(f'src/geone/_version.py', 'r') as f:
    exec(f.read())

setuptools.setup(
    name='geone',
    version=__version__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['geone', 'geone.deesse_core', 'geone.geosclassic_core'],
    package_dir={
        'geone':'src/geone',
        'geone.deesse_core':deesse_core_dir_selected,
        'geone.geosclassic_core':geosclassic_core_dir_selected
        },
    package_data={
        'geone.deesse_core':['*'],
        'geone.geosclassic_core':['*']
        },
    include_package_data=False, # False to prevent files from MANIFEST.in to be included in the wheel (.whl)
    # data_files=... # do not use! set files to be included in the source distribution (.tar.gz) in MANIFEST.in
    license=open('LICENSE', encoding='utf-8').read()
    # # already defined in pyproject.toml...
    # author="Julien Straubhaar",
    # author_email="julien.straubhaar@unine.ch",
    # description="Geostatistics tools and Multiple Point Statistics",
    # url='https://github.com/randlab/geone',
    # install_requires=['matplotlib', 'numpy', 'pandas', 'pyvista', 'scipy'],
)
