from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pycre", 
    version="0.0.1",
    author="Riccardo Cadei", "Naeem Khoshnevis", "Falco Joannes Bargagli Stoffi"
    author_email="rcadei@hsph.harvard.edu", "nkhoshnevis@g.harvard.edu", "fbargaglistoffi@hsph.harvard.edu"
    maintainer="Naeem Khoshnevis",
    maintainer_email="nkhoshnevis@g.harvard.edu",
    description="Python implementation of Causal Rule Ensemble algorithm.",
    long_description="Provides a new method for interpretable heterogeneous 
        treatment effects characterization in terms of decision rules 
        via an extensive exploration of heterogeneity patterns by an 
        ensemble-of-trees approach, enforcing high stability in the 
        discovery. It relies on a two-stage pseudo-outcome regression, and 
        theoretical convergence guarantees support it. Bargagli-Stoffi, 
        F. J., Cadei, R., Lee, K., & Dominici, F. (2023) Causal rule ensemble: 
        Interpretable Discovery and Inference of Heterogeneous Treatment Effects.  
        arXiv preprint <arXiv:2009.09036>.",
    long_description_content_type="text/markdown",
    url="TBD",
    license="GPL-3",
    packages=find_packages(exclude=['docs*', 'tests*']),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GPL3 License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    python_requires='>=3.7',
)
