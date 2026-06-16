# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""Configuration file for the Sphinx documentation builder."""

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'dpsynth'
copyright = '2026, Google LLC'  # pylint: disable=redefined-builtin
author = 'dpsynth authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',  # For markdown support
]

# templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']

# Source suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
