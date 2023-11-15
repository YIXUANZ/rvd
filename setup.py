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
                      'numpy==1.24.4', 
                      'scipy==1.10.1',
                      'SoundFile==0.12.1',
                      'librosa==0.9.2'                    
                      ],
    classifiers=['License :: OSI Approved :: MIT License'],
)