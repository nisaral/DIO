"""Single source of truth for the package version.

Kept in its own module with no imports so setuptools can read it statically
(``[tool.setuptools.dynamic] version = {attr = "dio._version.__version__"}``)
and other modules can import it without creating an import cycle.
"""

__version__ = "0.4.2"
