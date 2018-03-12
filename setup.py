
from setuptools import setup, find_packages
import os.path

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'annulus',
    version = '1.0.0',
    description = "Annulus (donut) detection in images for camera calibration",
    long_description = long_description, 

    url = "https://github.com/mgb4/Annulus",
    author = "Michael Burisch",
    author_email = "michael.burisch@gmx.com",
    
    keywords = "camera calibration marker detection",

    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ],
    
    #packages = find_packages(where = "annulus"),
    packages = find_packages(include = ["annulus"]),
)