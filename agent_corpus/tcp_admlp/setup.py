from setuptools import setup, find_packages

setup(
    name="drivora-tcp-admlp",
    version="0.1",
    packages=find_packages(include=["TCP*", "ADMLP*", "team_code*"]),
)
