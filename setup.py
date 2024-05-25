"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
license = (here / "LICENSE").read_text(encoding="utf-8")

setup(
    name='metaflow-nlp',
    version='0.1.0',
    description='Metaflow NLP toy project',
    long_description=long_description,
    url="https://github.com/dewith/metaflow-nlp/",
    author='Dewith Miramón',
    author_email='dewithmiramon@gmail.com',
    license=license,
    package_dir={"": "src"},
    packages=find_packages(where='src'),
)