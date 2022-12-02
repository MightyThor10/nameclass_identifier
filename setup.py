from setuptools import setup

setup(name='nameclass_identifier',
      version='0.1',
      description='A tool ',
      keywords='nameclass',
      url='http://github.com/',
      author='Elyas Bakhtiari',
      maintainer='Sayyed Hadi Razmjo',
      maintainer_email='example@example.com',
      author_email='example@example.com',
      license='MIT',
      packages=['nameclass_identifier'],
      install_requires=['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras'],
      python_requires='>=3',
      zip_safe=False)