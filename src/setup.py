from setuptools import setup, find_packages

setup(
    name="openne",
    url="https://github.com/thunlp/OpenNE",
    license="MIT",
    author="THUNLP",
    version="2.0.0",
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
    setup_requires=[]
)