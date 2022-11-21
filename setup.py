# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:13:49 2018

@author: MEASURE_A
"""

from setuptools import setup, find_packages

setup(name='soii_neural_autocoder',
      version='0.1',
      author='Alexander Measure',
      author_email='measure.alex@bls.gov',
      packages=find_packages(),
      include_package_data=True,
      install_requires=['tensorflow==2.9.3',
                        'nltk',
                        'pyyaml',
                        'pandas',
                        'keras==2.1.6',
                        'sklearn'])
