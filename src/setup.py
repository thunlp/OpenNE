from setuptools import setup, find_packages

setup(
    name="openne",
    url="https://github.com/Freyr-Wings/OpenNE",
    license="MIT",
    version='0.0.1',
    author="Freyr, Alan",
    author_email="alan1995wang@outlook.com",
    description="Open Source Network Embedding toolkit",
    packages= find_packages(),
    long_description=open("../README.md").read(),
    zip_safe=False,
    install_requires=[
        'gensim==3.0.1',
        'networkx==2.0',
        'pandas==0.23.4', 
        'scikit-learn==0.19.0',
        'tensorflow==1.10.0'
    ]
)