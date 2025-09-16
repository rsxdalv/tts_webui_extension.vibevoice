import setuptools

setuptools.setup(
    name="tts_webui_extension.vibevoice",
    packages=setuptools.find_namespace_packages(),
    version="0.2.1",
    author="rsxdalv",
    description="A template extension for TTS Generation WebUI",
    url="https://github.com/rsxdalv/tts_webui_extension.vibevoice",
    project_urls={},
    scripts=[],
    install_requires=[
        "vibevoice @ git+https://github.com/rsxdalv/vibevoice@stable",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

