from setuptools import find_packages, setup

DISTNAME = "xskillscore"
AUTHOR = "Ray Bell"
AUTHOR_EMAIL = "rayjohnbell0@gmail.com"
DESCRIPTION = "xskillscore"
LONG_DESCRIPTION = """Metrics for verifying forecasts"""
URL = "https://github.com/xarray-contrib/xskillscore"
with open("requirements.txt") as f:
    INSTALL_REQUIRES = f.read().strip().split("\n")
PYTHON_REQUIRE = ">=3.9"
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
    "Topic :: Scientific/Engineering :: Mathematics",
]

EXTRAS_REQUIRE = {
    "accel": ["numba>=0.52", "bottleneck"],
}
EXTRAS_REQUIRE["complete"] = sorted({v for req in EXTRAS_REQUIRE.values() for v in req})
# after complete is set, add in test
EXTRAS_REQUIRE["test"] = [
    "cftime",
    "matplotlib",
    "pre-commit",
    "pytest",
    "pytest-cov",
    "pytest-lazy-fixtures",
    "pytest-xdist",
    "scikit-learn",
]
EXTRAS_REQUIRE["docs"] = EXTRAS_REQUIRE["complete"] + [
    "doc8",
    "nbsphinx",
    "nbstripout",
    "sphinx",
    "sphinx-autosummary-accessors",
    "sphinx-copybutton",
    "sphinx-rtd-theme>=1.0",
    "sphinxcontrib-napoleon",
]

setup(
    name=DISTNAME,
    license_files = ('LICENSE.txt',),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
    url=URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    test_suite="xskillscore/tests",
    tests_require=["pytest"],
    python_requires=PYTHON_REQUIRE,
    use_scm_version={"version_scheme": "post-release", "local_scheme": "dirty-tag"},
    setup_requires=[
        "setuptools_scm",
        "setuptools>=30.3.0",
        "setuptools_scm_git_archive",
    ],
    extras_require=EXTRAS_REQUIRE,
    zip_safe=False,
)
