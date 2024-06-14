from setuptools import setup, find_packages

__version__ = "0.0.1"

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
        'aeon==0.6.0',
        'setuptools>=65.5.1',
        'numpy==1.24.4',
        'pandas==2.0.3',
        'matplotlib==3.7.2',
        'scipy==1.11.3',
        'scikit-learn==1.2.2',
        'networkx==3.1',
        ]
)