
from setuptools import setup

long_description = "Detection of annuli (donuts) in images for camera calibration."
	
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
    #packages = find_packages(include = ["annulus"]),
	packages = ["annulus"],
	python_requires='>=3',
)