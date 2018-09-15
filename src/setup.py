from setuptools import setup, find_packages

setup(
    name="openne",
    url="https://github.com/thunlp/OpenNE",
    license="MIT",
    author="THUNLP",
    description="Open Source Network Embedding toolkit",
    packages=find_packages(),
    long_description=open("../README.md").read(),
    zip_safe=False,
    setup_requires=[]
)