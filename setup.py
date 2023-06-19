from setuptools import find_packages, setup
from audio_preprocess import (
    __version__,
    __author__,
    __license__,
    __description__,
)
setup(
    name='audio_preprocess',
    packages=find_packages(include=['audio_preprocess', 'audio_preprocess.*']),
    version=__version__,
    description=__description__,
    author=__author__,
    license=__license__,
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='./tests/',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)