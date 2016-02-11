from setuptools import setup, find_packages

if __name__ == '__main__':
  name = 'ppca'
  setup(
    name         = name,
    version      = "0.0.2",
    author       = 'Allen Tran',
    author_email = 'realallentran@gmail.com',
    description  = 'Probabilistic PCA',
    packages     = find_packages(),
    classifiers  = [
      'Development Status :: 4 - Beta',
      'Programming Language :: Python',
      'Operating System :: Unix',
      'Operating System :: MacOS',
    ],
    setup_requires = [
      'setuptools>=3.4.4',
    ],
    install_requires = [
      'numpy',
      'scipy',
    ],


  )
