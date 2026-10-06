from setuptools import find_packages, setup

setup(
    name="tropical-reinforcement-learning",
    version="0.1.0",
    description="Verified fragment composition for language-model agents",
    packages=find_packages(include=["ragen", "ragen.*"]),
    python_requires=">=3.12",
)
