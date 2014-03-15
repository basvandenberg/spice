from distutils.core import setup

setup(
    name='SPiCE',
    version='0.1.2',
    author='B.A. van den Berg',
    author_email='b.a.vandenberg@gmail.com',
    packages=['spice', 'spice.plotpy', 'spice.job_runner'],
    scripts=['bin/featext', 'bin/classification', 'bin/classify', 'bin/job_runner'],
    url='http://pypi.python.org/pypi/SPiCE/',
    license='LICENSE.txt',
    description='Sequence-based Protein Classification and Exploration',
    long_description=open('README.md').read(),
)
