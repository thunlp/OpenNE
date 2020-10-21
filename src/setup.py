from setuptools import setup, find_packages
import readme_renderer
setup(
    name="openne",
    url="https://github.com/thunlp/OpenNE/tree/pytorch",
    license="MIT",
    author="THUNLP",
    version="1.0.1",
    description="Open Source Network Embedding toolkit",
    packages=find_packages(),
    long_description=open("../README.md").read(),
    long_description_content_type='text/markdown',
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.7',
    setup_requires=['torch>=1.5.0', 'six', 'numpy>=1.14',
                    'scipy>=0.19.1', 'gensim', 'scikit-learn>=0.19.0',
                    'networkx>=2.0', 'overloading']
)
