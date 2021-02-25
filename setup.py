from setuptools import find_packages, setup

DISTNAME = "xskillscore"
LICENSE = "Apache"
AUTHOR = "Ray Bell"
AUTHOR_EMAIL = "rayjohnbell0@gmail.com"
DESCRIPTION = "xskillscore"
LONG_DESCRIPTION = """Metrics for verifying forecasts"""
URL = "https://github.com/xarray-contrib/xskillscore"
with open("requirements.txt") as f:
    INSTALL_REQUIRES = f.read().strip().split("\n")
TESTS_REQUIRE = ["pytest"]
PYTHON_REQUIRE = ">=3.7"


setup(
    name=DISTNAME,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    test_suite="xskillscore/tests",
    tests_require=TESTS_REQUIRE,
    python_requires=PYTHON_REQUIRE,
    use_scm_version={"version_scheme": "post-release", "local_scheme": "dirty-tag"},
    setup_requires=[
        "setuptools_scm",
        "setuptools>=30.3.0",
        "setuptools_scm_git_archive",
    ],
    zip_safe=False,
)
