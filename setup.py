from setuptools import setup

setup(
    name='RVD',
    version='0.1.0',    
    description='Leveraging Laryngograph Data for Robust Voicing Detection in Speech',
    url='https://github.com/YIXUANZ/RVD',
    author='Yixuan Zhang, Heming Wang, and DeLiang Wang',
    author_email='zhang.7388@osu.edu',
    license='MIT',
    packages=['RVD'],
    install_requires=['mpi4py>=2.0',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)