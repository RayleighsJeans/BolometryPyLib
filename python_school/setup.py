from setuptools import setup, find_packages

setup(
    name='newton',
    version='0.0.1',
    description='mass to weight/weight to mass conversion',
    #download_url=url + '/tarball/' + 0.0.1,
    author='Philipp Hacker',
    license='GPL v3.0',
    packages=find_packages(),
    test_require=['nose'],
    # install_requires=['numpy', 'scipy'],
    entry_points={
        'console_scripts': ['newton = newton.app:main']
        }
    )
