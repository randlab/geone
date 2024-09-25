# For printing messages, use:
#    pip install . --verbose
# or
#    pip install . -v

import sys, os, platform, glob
import setuptools

src_dir = 'src'

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
    print(f'{COL_ERR}ERROR: getting config{COL_RESET}')
    exit()

print(f'{COL_CONFIG}Your configuration: platform_system={platform_system}, python_version={python_version}, glibc_version={glibc_version}{COL_RESET}')

# Set subdir_name: name of the subdirectory containing the C libraries,
# according to the platform system, the version of python and the version of glibc
# [subdir_name = <prefix>_<suffix>]

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
if python_version == (3, 6):
    suffix = 'py36'
elif python_version == (3, 7):
    suffix = 'py37'
elif python_version == (3, 8):
    suffix = 'py38'
elif python_version == (3, 9):
    suffix = 'py39'
elif python_version == (3, 10):
    suffix = 'py310'
elif python_version == (3, 11):
    suffix = 'py311'
elif python_version == (3, 12):
    suffix = 'py312'
else:
    suffix = None

# Set subdir_name
if prefix is None or suffix is None:
    subdir_name = None
else:
    subdir_name = f'{prefix}_{suffix}'

if subdir_name is None:
    print(f'{COL_ERR}ERROR: package geone not available for your configuration [platform_system={platform_system}, python_version={python_version}, glibc_version={glibc_version}]{COL_RESET}')
    exit()

# Set long_description
with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

# Load version
__version__ = '0.0.0' # default
with open(f'{src_dir}/geone/_version.py', 'r') as f:
    exec(f.read())

# Set directories containing the libraries and data files
src_dir = 'src'
pkg_name = 'geone'

lib_deesse_subdir = 'lib_deesse_core'
lib_deesse_platform_subdir_list = sorted(glob.glob('*', root_dir=f'{src_dir}/{pkg_name}/{lib_deesse_subdir}'))
lib_deesse_target_dir_list = [f'{src_dir}.{pkg_name}.{lib_deesse_subdir}.{s}' for s in lib_deesse_platform_subdir_list]
lib_deesse_files_list = [glob.glob(f'{src_dir}/{pkg_name}/{lib_deesse_subdir}/{s}/*', root_dir=None) for s in lib_deesse_platform_subdir_list]
deesse_core_dir_selected = f'{src_dir}/{pkg_name}/{lib_deesse_subdir}/{subdir_name}'

lib_geosclassic_subdir = 'lib_geosclassic_core'
lib_geosclassic_platform_subdir_list = sorted(glob.glob('*', root_dir=f'{src_dir}/{pkg_name}/{lib_geosclassic_subdir}'))
lib_geosclassic_target_dir_list = [f'{src_dir}.{pkg_name}.{lib_geosclassic_subdir}.{s}' for s in lib_geosclassic_platform_subdir_list]
lib_geosclassic_files_list = [glob.glob(f'{src_dir}/{pkg_name}/{lib_geosclassic_subdir}/{s}/*', root_dir=None) for s in lib_geosclassic_platform_subdir_list]
geosclassic_core_dir_selected = f'{src_dir}/{pkg_name}/{lib_geosclassic_subdir}/{subdir_name}'

data_files = list(zip(lib_deesse_target_dir_list, lib_deesse_files_list)) + list(zip(lib_geosclassic_target_dir_list, lib_geosclassic_files_list))

setuptools.setup(
    name='geone',
    version=__version__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['geone', 'geone.deesse_core', 'geone.geosclassic_core'],
    package_dir={
        'geone':f'{src_dir}/geone',
        'geone.deesse_core':deesse_core_dir_selected,
        'geone.geosclassic_core':geosclassic_core_dir_selected
        },
    package_data={
        'geone.deesse_core':['*'],
        'geone.geosclassic_core':['*']
        },
    include_package_data=True,
    data_files=data_files, # sources (lib) for all platforms
    license=open('LICENSE', encoding='utf-8').read()
    # # already defined in pyproject.toml...
    # author="Julien Straubhaar",
    # author_email="julien.straubhaar@unine.ch",
    # description="Geostatistics tools and Multiple Point Statistics",
    # url='https://github.com/randlab/geone',
    # install_requires=['matplotlib', 'numpy>=1,<2', 'pandas', 'pyvista', 'scipy'],
)
