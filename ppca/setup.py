from setuptools import setup, find_packages

if __name__ == '__main__':
  name = 'pca_missing'
  setup(
    name         = name,
    version      = "0.0.0",
    author       = 'Allen Tran',
    author_email = 'fakeallentran@gmail.com',
    description  = 'Probabilistic PCA for replacing missing values',
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
  )
