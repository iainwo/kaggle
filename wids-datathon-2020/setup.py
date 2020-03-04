from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='wids-datathon-2020',
    version='0.1.3',
    license='MIT',
    author='Iain Wong',
    author_email='iainwong@outlook.com',
    description='The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. (Kaggle Proj) https://www.kaggle.com/c/widsdatathon2020/overview',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/iainwo/kaggle/tree/master/wids-datathon-2020',
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
        'pyarrow',
        'numpy',
        'scikit-learn',
    ],
    python_requires='>=3.6',
)
