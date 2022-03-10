import sys, platform
import setuptools

with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

platform_system = platform.system()
python_version = sys.version_info[0:2]

if platform_system == 'Windows':
    if python_version == (3, 6):
        deesse_core_dir = 'geone/deesse_core/win_py36'
        geosclassic_core_dir = 'geone/geosclassic_core/win_py36'
    elif python_version == (3, 7):
        deesse_core_dir = 'geone/deesse_core/win_py37'
        geosclassic_core_dir = 'geone/geosclassic_core/win_py37'
    elif python_version == (3, 8):
        deesse_core_dir = 'geone/deesse_core/win_py38'
        geosclassic_core_dir = 'geone/geosclassic_core/win_py38'
    elif python_version == (3, 9):
        deesse_core_dir = 'geone/deesse_core/win_py39'
        geosclassic_core_dir = 'geone/geosclassic_core/win_py39'
    else:
        #print('pakcage geone not available for this python version ({}.{})'.format(*sys.version_info[0:2]))
        exit()
elif platform_system == 'Linux':
    if python_version == (3, 6):
        deesse_core_dir = 'geone/deesse_core/linux_py36'
        geosclassic_core_dir = 'geone/geosclassic_core/linux_py36'
    elif python_version == (3, 7):
        deesse_core_dir = 'geone/deesse_core/linux_py37'
        geosclassic_core_dir = 'geone/geosclassic_core/linux_py37'
    elif python_version == (3, 8):
        deesse_core_dir = 'geone/deesse_core/linux_py38'
        geosclassic_core_dir = 'geone/geosclassic_core/linux_py38'
    elif python_version == (3, 9):
        deesse_core_dir = 'geone/deesse_core/linux_py39'
        geosclassic_core_dir = 'geone/geosclassic_core/linux_py39'
    else:
        #print('pakcage geone not available for this python version ({}.{})'.format(*sys.version_info[0:2]))
        exit()
else:
    exit()

# Load version
with open('geone/_version.py', 'r') as f:
    exec(f.read())

setuptools.setup(
    name='geone',
    version=__version__,
    author="Julien Straubhaar",
    author_email="julien.straubhaar@unine.ch",
    description="Geostatistics simulation tools",
    long_description=long_description,
    install_requires=['matplotlib', 'numpy', 'pyvista', 'scipy'],
    packages=['geone', 'geone.deesse_core', 'geone.geosclassic_core'],
    package_dir={'geone':'geone', 'geone.deesse_core':deesse_core_dir, 'geone.geosclassic_core':geosclassic_core_dir},
    package_data={'geone.deesse_core':['*'], 'geone.geosclassic_core':['*']},
    include_package_data=True,
    license=open('LICENSE', encoding='utf-8').read()
)
