from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fatcat-gpt",
    version="0.2.0",
    description="胖貓專屬嘴臭LLM推理API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="caster97",
    packages=find_packages(),
    include_package_data=True, 
    package_data={
        "fatcat-gpt": ["*.json"],
    },
    install_requires=[
        "torch==2.8.0",
        "unsloth==2026.5.8",
        "peft==0.19.1",
        "transformers==4.56.2",
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
        "opencc-python-reimplemented"
    ],
)