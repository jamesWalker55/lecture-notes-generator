from pathlib import Path

# path of the main package
# call .parent twice, because this file is in lib/
PACKAGE_DIR = Path(__file__).absolute().parent.parent
# path to the repository holding the package
BASE_DIR = PACKAGE_DIR.parent

TESTS_DIR = BASE_DIR / "tests"
CACHE_DIR = BASE_DIR / ".cache"
