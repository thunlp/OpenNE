from setuptools import setup, find_packages

setup(
    name="openne",
    url="https://github.com/Freyr-Wings/OpenNE",
    license="MIT",
    author="Freyr, Alan",
    author_email="alan1995wang@outlook.com",
    description="Open Source Network Embedding toolkit",
    packages=find_packages(),
    long_description=open("../README.md").read(),
    zip_safe=False,
    setup_requires=[]
)