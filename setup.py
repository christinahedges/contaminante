#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("python setup.py bdist_wheel")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/contaminante*")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('contaminante/version.py').read())

# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
# 2. What dependencies required to run the unit tests? (i.e. `pytest --remote-data`)
tests_require = ['pytest', 'pytest-cov', 'pytest-remotedata', 'codecov', 'pytest-doctestplus', 'codacy-coverage']
# 3. What dependencies are required for optional features?

setup(name='contaminante',
      version=__version__,
      description="Find the contaminant transiting source in Kepler, K2 or TESS data. ",
      long_description=open('README.md').read(),
      author='Christina Hedges',
      author_email='christina.l.hedges@nasa.gov',
      license='MIT',
      package_dir={
            'contaminante': 'contaminante'},
      packages=['contaminante'],
      install_requires=install_requires,
      setup_requires=['pytest-runner'],
      tests_require=tests_require,
      include_package_data=True,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
