from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
        'opencv-python==4.0.0.21',
        'h5py<3.0.0'
        ]

setup(
    name='backgroundremoval',
    version='1.1',
    description='This application performs the background removal for GarmentCleanup garment images.',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    scripts=['predictor.py', 'tools.py', 'defaults.py']
)
