from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='halite-agent',
    version='0.1.0',
    license='MIT',
    author='Iain Wong',
    author_email='iainwong@outlook.com',
    description='The challenge is to create an agent that can succeed in the game of Halite IV.  (Kaggle Proj) https://www.kaggle.com/c/halite',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/iainwo/kaggle/tree/master/halite',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pandas',
        'python-dotenv',
        'Click',
    ],
    python_requires='>=3.6',
)
