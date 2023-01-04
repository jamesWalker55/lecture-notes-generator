from setuptools import setup

setup(
    name="Lecture Notes Generator",
    version="0.1",
    description="Generate notes for any slideshow-based lecture videos.",
    author="James Walker",
    author_email="james.chunho@gmail.com",
    license="MIT",
    packages=["lecture_notes_generator"],
    install_requires=[
        "opencv-python",
        "tqdm",
    ],
    zip_safe=False,
)
