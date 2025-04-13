from setuptools import setup, find_packages

setup(
    name="llmbench",
    version="0.1.0",
    description="A tool to compare LLM models and their benchmark scores",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Abhigna Nagaraja",
    author_email="abhigna.n4@gmail.com",
    url="https://github.com/abhigna/llmbench",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "flask",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "llmbench=llmbench.cli:main",
            "update-benchmarks=llmbench.update:main",
        ],
    },
    include_package_data=True,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # This ensures the data, static, and templates directories are included
    package_data={
        "": ["../../static/**/*", "../../templates/**/*", "../../data/**/*"],
    },
)