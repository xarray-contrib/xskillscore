# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

import xskillscore

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
current_year = datetime.datetime.now().year
project = 'xskillscore'
copyright = f'2020-{current_year}, Ray Bell'
author = 'Ray Bell'

# The full version, including alpha/beta/rc tags
version = xskillscore.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.napoleon',
    'sphinx.ext.imgmath',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
]

extlinks = {
    'issue': ('https://github.com/raybellwaves/xskillscore/issues/%s', 'GH#'),
    'pr': ('https://github.com/raybellwaves/xskillscore/pull/%s', 'GH#'),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'
source_suffix = '.rst'
master_doc = 'index'
nbsphinx_timeout = 60


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# Can add below once we have a logo.
# html_logo = 'images/esmtools-logo.png'
# html_theme_options = {'logo_only': True, 'style_nav_header_background': '#fcfcfc'}
