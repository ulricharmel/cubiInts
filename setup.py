import os
from setuptools import setup, find_packages

pkg = 'cubiints'
__version__ = "0.0"
build_root = os.path.dirname(__file__)

def readme():
	"""Get readme content for package long description"""
	with open(os.path.join(build_root, 'README.rst')) as f:
		return f.read()


def requirements():
	"""Get package requirements"""
	with open(os.path.join(build_root, 'requirements.txt')) as f:
		return [pname.strip() for pname in f.readlines()]


setup(name=pkg,
      version=__version__,
      description="A tool for searching optimal solution intervals for the CubiCal package.",
      long_description=readme(),
      author="Ulrich A. Mbou Sob",
      author_email="mulricharmel@gmail.com",
      packages=find_packages(),
      # url="https://github.com/Athanaseus/aimfast",
      license="GNU GPL 3",
      classifiers=["Development Status :: 4 - Beta",
                   "Intended Audience :: Developers",
                   "Programming Language :: Python :: 3.6",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Software Development :: Libraries :: Python Modules"],
      keywords="fits dataset lsm statistics models html jupyter",
      platforms=["OS Independent"],
      install_requires=requirements(),
      # tests_require=["attrs",
      #                "pytest",
      #                "numpy"],
      # extras_require={'docs': ["sphinx-pypi-upload",
      #                          "numpydoc",
      #                          "Sphinx"],
      #                 'aegean': ["AegeanTools"],
      #                 'bdsf': ["bdsf", "matplotlib"],
      #                 'source_finders': ["bdsf",
      #                                    "AegeanTools"]},
      python_requires='>=3.6',
      include_package_data=True,
      scripts=['cubiints/bin/cubiints'])