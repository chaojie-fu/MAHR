import os

from setuptools import setup, find_packages

# Read project dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# handle dependencies under `third_party` directory
third_party_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
for dep in os.listdir(third_party_dir):
    dep_path = os.path.join(third_party_dir, dep)
    pyproject_toml = os.path.join(dep_path, "pyproject.toml")
    # only supports third party dependency with pyproject.toml file included
    if os.path.isfile(pyproject_toml):
        requirements.append(f"{dep} @ file://{dep_path}")

# noinspection PyTypeChecker
setup(
    name="mocap_data_converter",
    version="0.1.0",
    packages=find_packages(include=["mocap_data_converter"]),  # Limit main module precisely
    install_requires=requirements,  # Auto-process from requirements.txt
    author="FU, Chaojie",
    description="Motion Capture Data Conversion Toolkit",
    python_requires=">=3.10",
    # Add other metadata fields as needed
)
