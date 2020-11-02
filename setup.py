from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
        'opencv-python==4.0.0.21'
        ]

setup(
    name='backgroundremoval',
    version='1.0',
    description='This application performs the background removal for Micolet garment images.',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    scripts=['predictor.py', 'tools.py', 'defaults.py']
)
