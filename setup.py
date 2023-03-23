from setuptools import setup, find_packages

setup(
    name="TrojanArmor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "scikit-learn>=0.24.0",
    ],
    author="Narek Maloyan",
    author_email="maloyan.narek@example.com",
    description="A library for experimenting with trojan attacks in neural networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maloyan/TrojanArmor",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="neural networks, trojan attacks, deep learning",
)

