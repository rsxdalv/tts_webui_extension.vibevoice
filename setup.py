import setuptools

setuptools.setup(
    name="extension_vibevoice",
    packages=setuptools.find_namespace_packages(),
    version="0.0.1",
    author="rsxdalv",
    description="A template extension for TTS Generation WebUI",
    url="https://github.com/rsxdalv/extension_vibevoice",
    project_urls={},
    scripts=[],
    install_requires=[
        "vibevoice @ git+https://github.com/rsxdalv/vibevoice@stable",
        # "chatterbox-tts @ git+https://github.com/rsxdalv/chatterbox@faster",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
