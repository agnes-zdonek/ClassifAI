from setuptools import setup, find_packages

setup(
    name='ClassifAi',
    version='1.2.0',
    author='AZ',
    description='A Python library for understanding the implementation of machine learning classifiers.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib'
    ]
)
