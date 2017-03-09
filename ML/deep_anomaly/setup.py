from setuptools import setup

setup(
    name                 = 'deep_anomaly',
    version              = '0.1',
    description          = 'Anomaly and outlier detection framework for Python.',
    long_description     = readme(),
    classifiers          = ['Development Status :: 3 - Alpha',
                            'License :: OSI Approved :: MIT License',
                            'Programming Language :: Python :: 2.7', 
                            'Topic :: Scientific/Engineering :: Artificial Intelligence',
                           ],
    url                  = 'http://github.com/pragaashp/deep_anomaly',
    author               = 'Pragaash Ponnusamy',
    author_email         = 'pvirgo.revo93@gmail.com',
    license              = 'MIT',
    packages             = ['deep_anomaly'],
    install_requires     = ['numpy', 'scipy', 'numba'],
    include_package_data = True,
    zip_safe             = False
)

def readme():
    with open('README.rst', 'r') as f:
        return f.read()