import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='traffic-flow',
    version='0.2.3',
    author='Peter Vanya',
    author_email='peter.vanya@gmail.com',
    license='GPLv3',
    description='Macroscopic transport modelling; forecasting traffic flows on roads',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/petervanya/traffic-flow',
    keywords='traffic flow modelling forecast optimisation transport',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.9',
    packages=setuptools.find_packages(),
    package_data={
        '': ['examples/network*.xlsx']
    },
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10.1',
        'pandas>=2.0.3',
        'geopandas>=0.12.2',
        'openpyxl>=3.1.2',
        'networkx>=3.2.1',
        'python-igraph>=0.11.3',
    ],
)
