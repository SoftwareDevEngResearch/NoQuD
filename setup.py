from codecs import open
from os import path
import sys

#Note: this file is a work in progress.

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'noqud', '_version.py')) as version_file:
    exec(version_file.read())

with open(path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'CHANGELOG.md')) as changelog_file:
    changelog = changelog_file.read()

desc = readme + '\n\n' + changelog
try:
    import pypandoc
    long_description = pypandoc.convert_text(desc, 'rst', format='md')
    with open(path.join(here, 'README.rst'), 'w') as rst_readme:
        rst_readme.write(long_description)
except (ImportError, OSError, IOError):
    long_description = desc

install_requires = [
    'numpy',
    'numpy',
    'pandas',
    'matplotlib',
    'numba',
]

tests_require = [
    'pytest',
    'pytest-cov',
]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setup(
    name='noqud',
    version=__version__,
    description='Step Characteristic and QD Nodal Solver for neutron transport',
    long_description=long_description,
    author='Aaron J Reynolds',
    author_email='reynolaa@oregonstate.edu',
    #url='http://physics.codes',
    classifiers=[
        'Intended Audience :: Nuclear Engineering',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
    ],
    license='BSD-3-Clause',
    install_requires=install_requires,
    tests_require=tests_require,
    python_requires='2.7',
    setup_requires=setup_requires,
    zip_safe=False,
    packages=find_packages()
    include_package_data=True,
)

if __name__ == "__main__":
    setup()