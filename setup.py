from setuptools import setup, find_packages


dependencies = [
    'ml-control @ git+https://github.com/HenKlei/REDUCED-OPT-CONTROL.git',
]

setup(
    name='adaptive-ml-control',
    version='0.1.0',
    description='Python implementation of an adaptive model hierarchy for optimal control of parameter-dependent systems',
    author='Hendrik Kleikamp',
    author_email='hendrik.kleikamp@uni-muenster.de',
    maintainer='Hendrik Kleikamp',
    maintainer_email='hendrik.kleikamp@uni-muenster.de',
    packages=find_packages(),
    install_requires=dependencies,
)
