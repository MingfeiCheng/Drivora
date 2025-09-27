from setuptools import setup, find_packages

setup(
    name="drivora-plant",
    version="0.1",
    packages=find_packages(include=["training*", "carla_agent_files*"]),
)
