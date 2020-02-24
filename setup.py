import sys, platform
import setuptools

with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

platform_system = platform.system()
python_version = sys.version_info[0:2]

if platform_system == 'Windows':
    if python_version == (3, 6):
        deesse_core_dir = 'geone/deesse_core/win_py36'
    elif python_version == (3, 7):
        deesse_core_dir = 'geone/deesse_core/win_py37'
    elif python_version == (3, 8):
        deesse_core_dir = 'geone/deesse_core/win_py38'
    else:
        exit()
elif platform_system == 'Linux':
    if python_version == (3, 6):
        deesse_core_dir = 'geone/deesse_core/linux_py36'
    elif python_version == (3, 7):
        deesse_core_dir = 'geone/deesse_core/linux_py37'
    elif python_version == (3, 8):
        deesse_core_dir = 'geone/deesse_core/linux_py38'
    else:
        exit()
else:
    exit()

setuptools.setup(
    name='geone',
    version='0.1.0',
    author="Julien Straubhaar",
    author_email="julien.straubhaar@unine.ch",
    description="Geostatistics simulation tools",
    long_description=long_description,
    install_requires=['numpy'],
    packages=['geone', 'geone.deesse_core'],
    package_dir={'geone':'geone', 'geone.deesse_core':deesse_core_dir},
    package_data={'geone.deesse_core':['*']},
    include_package_data=True,
    license=open('LICENSE').read()
)
