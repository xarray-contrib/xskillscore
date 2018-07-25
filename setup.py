from setuptools import find_packages, setup


DISTNAME = 'xskillscore'
VERSION = '0.0.1'
LICENSE = 'Apache'
AUTHOR = 'Ray Bell'
AUTHOR_EMAIL = 'rjbell1987@gmail.com'
DESCRIPTION = "xskillscore"
URL = 'https://github.com/pydata/xarray'
INSTALL_REQUIRES = ['scikit-learn', 'xarray','dask']
TESTS_REQUIRE = ['pytest']
PYTHON_REQUIRE = '>=3.6'


setup(name=DISTNAME,
      version=VERSION,
      license=LICENSE,
      author=AUTHOR,      
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      url=URL,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      python_requires=PYTHON_REQUIRE)

