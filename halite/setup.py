from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='src',
    version='0.1.0',
    license='MIT',
    author='Iain Wong',
    author_email='iainwong@outlook.com',
    description='Created by Two Sigma in 2016, more than 15,000 people around the world have participated in a Halite challenge. Players apply advanced algorithms in a dynamic, open source game setting. The strategic depth and immersive, interactive nature of Halite games make each challenge a unique learning environment.',
    long_description=long_description,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)