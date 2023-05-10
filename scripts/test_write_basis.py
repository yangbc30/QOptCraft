from sys import path
from os.path import join, abspath

module_path = abspath(join(".."))
if module_path not in path:
    path.append(module_path)

from QOptCraft import write_algebra_basis


if __name__ == "__main__":
    n = 3
    m = 4 + (n - 2)
    write_algebra_basis(3, [n] + [0] * (m - 1), False)
