from setuptools import setup, find_namespace_packages

setup(
    name="my_modules",
    version="0.1",
    packages=find_namespace_packages(where="scripts"),
    package_dir={"": "scripts"},
)