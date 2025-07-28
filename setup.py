from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="custo-clarity",
    version="1.0.0",
    author="Neelanjan Chakraborty",
    author_email="contact@neelanjanchakraborty.in",
    description="Customer Segmentation Analysis for Retail Strategy using Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Neelanjan-chakraborty/custo-clarity",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Data Scientists",
        "Intended Audience :: Business Analysts",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="customer-segmentation, machine-learning, clustering, retail-analytics, data-science",
    project_urls={
        "Bug Reports": "https://github.com/Neelanjan-chakraborty/custo-clarity/issues",
        "Source": "https://github.com/Neelanjan-chakraborty/custo-clarity",
        "Documentation": "https://github.com/Neelanjan-chakraborty/custo-clarity/docs",
        "Author Website": "https://neelanjanchakraborty.in/",
    },
)
