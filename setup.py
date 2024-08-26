from setuptools import setup, find_packages
import setuptools

import warnings
warnings.filterwarnings("ignore")

# Disable version normalization performed by setuptools.setup()
try:
    # Try the approach of using sic(), added in setuptools 46.1.0
    from setuptools import sic
except ImportError:
    # Try the approach of replacing packaging.version.Version
    sic = lambda v: v
    try:
        # setuptools >=39.0.0 uses packaging from setuptools.extern
        from setuptools.extern import packaging
    except ImportError:
        # setuptools <39.0.0 uses packaging from pkg_resources.extern
        from pkg_resources.extern import packaging
    packaging.version.Version = packaging.version.LegacyVersion

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="paani",
    version=sic("0.1.0"),
    url="https://github.com/cloudjo21/paani.git",
    packages=find_packages("src"),
    package_dir={"paani": "src/paani"},
    python_requires=">=3.11.6",
    long_description=open("README.md").read(),
    install_requires=required,
    normalize_version=False
)

