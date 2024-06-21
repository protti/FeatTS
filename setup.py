from setuptools import setup, find_packages

__version__ = "0.0.3"

setup(
    name="FeatTS",
    version=__version__,
    description="A new method for clustering time series by adopting the best statistical features.",
    author="Donato Tiano",
    author_email="donatotiano@gmail.com",
    packages=find_packages(),
    zip_safe=True,
    license="",
    url="https://github.com/protti/FeatTS",
    entry_points={},
    install_requires=[
        'fastdtw>=0.3.4',
        'networkx>=2.5.1',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'pyclustering==0.10.1.2',
        'scikit_learn>=0.24.2',
        'scipy==1.13.1',
        'tsfresh>=0.18.0',
        'aeon>=0.6.0'
        ]
)