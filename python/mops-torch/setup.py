import os
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext

ROOT = os.path.realpath(os.path.dirname(__file__))


class cmake_ext(build_ext):
    """Build the native library using cmake"""

    def run(self):
        import torch

        source_dir = os.path.join(ROOT, "lib")
        build_dir = os.path.join(ROOT, "build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "mops", "torch")

        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ]

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                build_dir,
                "--parallel",
                "--config",
                "Release",
                "--target",
                "install",
            ],
            check=True,
        )


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs.\nUse `pip install .` or "
            "`python -m build --wheel . && pip install dist/mops_torch-*.whl` "
            "to install from source."
        )


if __name__ == "__main__":
    with open(os.path.join(ROOT, "VERSION")) as fd:
        version = fd.read().strip()

    setup(
        version=version,
        ext_modules=[
            Extension(name="mops_torch", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
        },
        package_data={
            "mops-torch": [
                "mops/torch/bin/*",
                "mops/torch/lib/*",
                "mops/torch/include/*",
            ]
        },
    )
