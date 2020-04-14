from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
        'opencv-python==4.0.0.21',
        'requests==2.22.0'
        ]

setup(
    name='micolet',
    version='0.1',
    description='This application performs the image correction for Micolet garment images.',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    scripts=['predictor.py', 'tools.py']
)
