import setuptools

__author__ = 'WangHao'
__version__ = '0.0.1'


def get_description():
    return ("Provide a simple way to get backtesting results without\
            need to care about the specifications")

with open("README.md","r") as fh:
    long_description=fh.read()


setuptools.setup(
    name="ptbt",
    version=__version__,
    author="WangHao",
    author_email="wanghao0524@outlook.com",
    description=get_description(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD",
    keywords="Portfolio backtesting",
    url="https://github.com/WilliamWang1996/ptbt",
    include_package_data=True,
    packages=setuptools.find_packages(),
    platforms=['any'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires='>=3.6',
)
