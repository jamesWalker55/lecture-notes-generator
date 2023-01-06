from setuptools import setup

setup(
    name="Lecture Notes Generator",
    version="0.1",
    description="Generate notes for any slideshow-based lecture videos.",
    author="James Walker",
    author_email="james.chunho@gmail.com",
    license="MIT",
    # Let setuptools detect packages automatically
    # https://setuptools.pypa.io/en/latest/userguide/package_discovery.html
    # packages=["lecture_notes_generator"],
    install_requires=[
        "opencv-python",
        "tqdm",
        "whisper @ git+https://github.com/openai/whisper.git",
        "Jinja2",
        "webvtt-py",
        "scipy",
    ],
    zip_safe=False,
)
