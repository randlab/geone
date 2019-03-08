import platform
import setuptools

with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

if platform.system() == 'Windows':
   deesse_lib = ['_deesse.pyd', 'rlm1212.dll', 'vcruntime140.dll']
else:
   deesse_lib = ['_deesse.so']

setuptools.setup(
    name='geone',
    version='0.1.0',
    author="Julien Straubhaar",
    author_email="julien.straubhaar@unine.ch",
    description="Geostatistics simulation tools",
    long_description=long_description,
    packages=setuptools.find_packages(),
    package_data={'geone.deesse_core':deesse_lib},
    include_package_data=True,
    license=open('LICENSE').read()
)
