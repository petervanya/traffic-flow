import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='traffic-flow',
    version='0.2.2',
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
    python_requires='>=3.7',
    packages=setuptools.find_packages(),
    package_data={
        '': ['examples/network*.xlsx']
    },
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.6',
        'pandas>=1.2, <2.0',
        'geopandas>=0.10.2',
        'openpyxl>=3.0.10',
        'networkx>=2.5',
        'python-igraph>=0.8, <0.10',
    ],
)
