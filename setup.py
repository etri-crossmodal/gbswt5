# -*- coding: utf-8 -*-
"""
    gbswt5 setup.py
"""
from setuptools import setup, find_packages


setup(
    name="gbswt5",
    version="0.1.0",
    description="GBST-KEByT5 language model(based on CharFormer(Tay et al., 2022)) implementation "
                "for huggingface Transformers.",
    author="Jong-hun Shin",
    author_email="luna.jetch@gmail.com",
    url="https://github.com/dalgarak/gbswt5",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(include=["gbswt5", "gbswt5.*"]),
    install_requires=['transformers>=4.27.0', 'torch>=1.11.0', 'einops>=0.6.0'],
    platforms='any',
)
