"""Copyright 2021 Daniel Gómez Aguado

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import sys
import warnings

import numpy as np


def read_matrix_from_txt(filename):
    print(f"\nSearching for file '{filename}.txt' in the directory...\n")

    try:
        matrix_file = open(f"{filename}.txt")

        # Loading S
        S = np.loadtxt(matrix_file, delimiter=",", dtype=complex)

    except FileNotFoundError:
        print("\nThe file required does not exist.\n")
        print(
            f"\nA {filename} matrix must be generated first.\n"
        )  # In our case of interest, with the main algorthms 1/1a
        print("\nThe program will end.\n")
        sys.exit()

    print("\nMatrix loading executed perfectly.\n")

    return S


# Function for reading of files with multiple variables: 'HEAD' and 'END' mark each's initiation and ending
# https://stackoverflow.com/questions/21993304/how-to-read-two-arrays-or-matrices-from-a-text-file-in-python
# exists to support the following function
def tokenizer(filename):
    with open(filename + ".txt") as f:
        chunk = []
        separator = []
        count = 0
        for line in f:
            if "*" in line:
                # print(count)
                separator.append(str(count))
                continue
            if "HEAD" in line:
                count += 1
                continue
            if "END" in line:
                yield chunk
                chunk = []
                continue
            chunk.append(line)
    yield separator


def read_matrix_from_txt_general(filename):
    print(f"\nSearching for file '{filename}.txt' in the directory...\n")

    try:
        warnings.filterwarnings("ignore")
        arrays = [np.loadtxt(A, delimiter=",", dtype=complex) for A in tokenizer(filename)]
        # arrays = [np.loadtxt(A,delimiter=',',dtype=complex) for A in tokenizer(filename)]

    except FileNotFoundError:
        print("\nThe file required does not exist.\n")
        print(
            f"\nA {filename} file must be generated first.\n"
        )  # In our case of interest, with the main algorthms 1/1a
        print("\nThe program will end.\n")
        sys.exit()

    print("\nMatrix loading executed perfectly.\n")

    return arrays[: (len(arrays) - 1)], np.array(arrays[len(arrays) - 1], dtype=int)
