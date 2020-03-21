from setuptools import find_packages, setup

DISTNAME = 'xskillscore'
VERSION = '0.0.14'
LICENSE = 'Apache'
AUTHOR = 'Ray Bell'
AUTHOR_EMAIL = 'rayjohnbell0@gmail.com'
DESCRIPTION = 'xskillscore'
LONG_DESCRIPTION = """Metrics for verifying forecasts"""
URL = 'https://github.com/raybellwaves/xskillscore'
INSTALL_REQUIRES = [
    'scikit-learn',
    'xarray',
    'dask',
    'scipy',
    'properscoring',
    'numba',
    'bottleneck'
]
TESTS_REQUIRE = ['pytest']
PYTHON_REQUIRE = '>=3.6'


setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    python_requires=PYTHON_REQUIRE,
)
