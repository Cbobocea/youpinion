from setuptools import setup, find_packages

setup(
    name="your_package_name",  # Replace with your package name
    version="0.1",  # Replace with the version of your package
    packages=find_packages(where="src"),  # Finds all packages under the "src" directory
    package_dir={"": "src"},  # Specifies that the packages are under the "src" directory
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    package_data={"mypkg": ["*.txt", "*.rst"]},  # Include text and rst files in the "mypkg" package
    exclude_package_data={"mypkg": [".gitattributes"]},  # Exclude specific files
)
