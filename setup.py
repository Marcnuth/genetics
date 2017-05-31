'''
Setup Genetics
'''

from setuptools import setup, find_packages


setup(
    name='genetics',
    version='1.0.0',

    description=("Genetic Algorithm in Python, which could be used for Sampling, Feature Select, Model Select, etc in Machine Learning"),

    url="https://github.com/Marcnuth/genetics",

    # Author details
    author="Marcnuth",
    author_email="marcnuth@foxmail.com",

    license="Apache License 2.0",

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3'

    ],

    # What does your project relate to?
    keywords=("genetic-algorithm, machine-learning, sampling, feature-engineering, model-selection"),

    packages=find_packages("./"),

    install_requires=["numpy"],
)
