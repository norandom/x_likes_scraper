from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="x-likes-exporter",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Export your liked tweets from X (Twitter) to JSON, Pandas, and Markdown",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/x-likes-exporter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "beautifulsoup4>=4.12.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "python-dateutil>=2.8.2",
    ],
    entry_points={
        "console_scripts": [
            "x-likes-export=cli:main",
        ],
    },
)
