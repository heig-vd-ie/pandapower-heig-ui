import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pandapower-heig-ui",                     # This is the name of the package
    version="0.7.1",                        # The initial release version
    author="Luca Tomasini",                     # Full name of the author
    description="A set of function coded in order to help heig-vd students to use pandapower library.",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="src"),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.10',                # Minimum version requirement of the package
    py_modules=["pp_heig_plot", "pp_heig_simulation", "pp_heig_signal"],             # Name of the python package
    package_dir={'':'src'},     # Directory of the source code of the package
    install_requires=[]                     # Install other dependencies if any
)