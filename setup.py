from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='wildfireassessment',
    version='0.1.0',
    description='Package for wildfire damage assessment notebook',
    long_description=readme,
    author='Ai-Linh Alten',
    author_email='ai-linh.alten@sjsu.edu',
    url='https://github.com/aalten77/wildfireassessment',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)