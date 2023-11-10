from setuptools import setup

setup(
    name='rvd',
    version='0.1.0',    
    description='Leveraging Laryngograph Data for Robust Voicing Detection in Speech',
    url='https://github.com/YIXUANZ/rvd',
    author='Yixuan Zhang, Heming Wang, and DeLiang Wang',
    author_email='zhang.7388@osu.edu',
    license='MIT',
    packages=['rvd'],
    package_data={'rvd': ['pretrained/*']},
    install_requires=['torch',
                      'numpy', 
                      'scipy',
                      'SoundFile',
                      'librosa<0.10'                    
                      ],
    classifiers=['License :: OSI Approved :: MIT License'],
)