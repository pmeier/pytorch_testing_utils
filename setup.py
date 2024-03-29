import subprocess
from os import path
from setuptools import find_packages, setup

PROJECT_ROOT = path.abspath(path.dirname(__file__))
PACKAGE_NAME = "pytorch_testing_utils"
PACKAGE_ROOT = path.join(PROJECT_ROOT, PACKAGE_NAME)


about = {}
with open(path.join(PACKAGE_ROOT, "__about__.py"), "r") as fh:
    exec(fh.read(), about)

with open(path.join(PROJECT_ROOT, "README.rst"), "r") as fh:
    long_description = fh.read()


class Git:
    def run(self, *cmds, cwd=None):
        return subprocess.check_output(("git", *cmds), cwd=cwd).decode("utf-8").strip()

    def is_available(self) -> bool:
        try:
            self.run("--help")
            return True
        except subprocess.CalledProcessError:
            return False

    def is_repo(self, dir: str) -> bool:
        return path.exists(path.join(dir, ".git"))

    def hash(self, dir: str) -> str:
        return self.run("rev-parse", "--short", "HEAD", cwd=dir)

    def is_dirty(self, dir: str) -> bool:
        return bool(self.run("status", "-uno", "--porcelain", cwd=dir))


def get_version():
    __version__ = about["__base_version__"]

    if not about["__is_dev_version__"]:
        return __version__
    __version__ += "+dev"

    git = Git()
    if not (git.is_available() and git.is_repo(PROJECT_ROOT)):
        return __version__

    __version__ += f".{git.hash(PROJECT_ROOT)}"

    if not git.is_dirty(PROJECT_ROOT):
        return __version__
    __version__ += ".dirty"

    return __version__


__version__ = get_version()
version_file = "__version__"

with open(path.join(PACKAGE_ROOT, version_file), "w") as fh:
    fh.write(__version__)
package_data = {PACKAGE_NAME: [version_file, "py.typed"]}

install_requires = ("numpy >= 1.5", "pytest")

classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Typing :: Typed",
]

setup(
    name=about["__name__"],
    description=about["__description__"],
    keywords=about["__keywords__"],
    version=__version__,
    url=about["__url__"],
    license=about["__license__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages(where=PROJECT_ROOT, exclude=("docs", "tests", ".github")),
    package_data=package_data,
    python_requires=">=3.6",
    install_requires=install_requires,
    classifiers=classifiers,
)
