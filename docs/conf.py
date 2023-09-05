# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../src'))

project = 'pandapower HEIG-VD user interface'
copyright = '2023, HEIG-VD IE'
author = 'HEIG-VD IE'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# More information about themes: https://cerodell.github.io/sphinx-quickstart-guide/build/html/theme.html
# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
# Why You Shouldn’t Use "Markdown" for Documentation:
# https://www.ericholscher.com/blog/2016/mar/15/dont-use-markdown-for-technical-docs/

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "myst_parser",
    "sphinx_copybutton",
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and directories to ignore when looking
# for source files. This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# The theme to use for HTML and HTML Help pages.  See the documentation for a list of builtin themes
# More themes can be found there: https://sphinx-themes.org/#themes
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
# html_theme = 'sphinx_book_theme'
html_theme = 'piccolo_theme'
html_theme_options = {
    # 'logo_only': True,
    # 'display_version': True,
    # 'prev_next_buttons_location': 'bottom',
    # 'style_external_links': True,
    # Toc options
    # 'collapse_navigation': False,
    # 'sticky_navigation': False,
    # 'navigation_depth': 3,
    # 'includehidden': True,
    # 'titles_only': False
}
# To generate a GitHub link into your SPHINX documentation:
# https://stackoverflow.com/questions/62904172/how-do-i-replace-view-page-source-with-edit-on-github-links-in-sphinx-rtd-th
# html_context = {
#     "display_github": True, # Integrate GitHub
#     "github_user": "MyUserName", # Username
#     "github_repo": "MyDoc", # Repo name
#     "github_version": "master", # Version
#     "conf_py_path": "/source/", # Path in the checkout to the docs root
# }
html_title = 'HEIG-VD IE Package Documentation'
# html_logo = 'path/to/logo.png'
# html_favicon = 'path/to/favicon.ico'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
