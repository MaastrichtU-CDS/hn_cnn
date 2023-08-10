from os import path
from codecs import open
from setuptools import setup, find_packages

# get current directory
here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# # read the API version from disk
# with open(path.join(here, 'vantage6', 'tools', 'VERSION')) as fp:
#     __version__ = fp.read()

# setup the package
setup(
    name='hn_cnn',
    version="1.0.0",
    description='Head and Neck CT Convolutional Neural Network',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pedro-cmat/hn_cnn',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy==1.21.4',
        'pandas==1.3.4',
        'Pillow==8.4.0',
        'scikit-learn==1.0.1',
        'scipy==1.7.3',
        'torch==1.10.0',
        'torchsummary==1.5.1',
        'torchvision==0.11.1'
    ]
    # ,
    # extras_require={
    # },
    # package_data={
    #     'vantage6.tools': [
    #         'VERSION'
    #     ],
    # }
)
