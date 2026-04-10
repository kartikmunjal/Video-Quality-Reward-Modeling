from setuptools import setup, find_packages

setup(
    name="vqrm",
    version="0.1.0",
    description="Video Quality Reward Modeling — benchmarking automated metrics against human preference",
    author="kartikmunjal",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0",
        "decord>=0.6.0",
        "opencv-python>=4.9.0",
        "lpips>=0.1.4",
        "einops>=0.7.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0.1",
    ],
)
