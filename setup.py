#!/usr/bin/env python
from setuptools import setup

import glider

description = ('Glider implements a lightweight (data) frame object based on '
               'numpy')

long_description= '''
The project goal is to provide a data-frame object based on numpy and
that acts as a complement to it.

Numpy already provides a large set of operations for array
manipulation, so Glider role is to combine those operations in order
to bring features that make sense on sets of arrays like join or
groupby.
'''

setup(name='Glider',
      version=glider.__version__,
      description=description,
      # long_description=long_description,
      author='Bertrand Chenal',
      author_email='bertrand@adimian.com',
      url='https://bitbucket.org/bertrandchenal/glider',
      install_requires=[
          'numpy>=1.14',
      ],
      license='MIT',
      py_modules=['glider'],
  )
