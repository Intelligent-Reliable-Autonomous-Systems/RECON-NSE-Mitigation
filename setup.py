from setuptools import setup, find_packages

setup(
    name='salp_domain',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'seaborn',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'simple_colors',
    ],
)

