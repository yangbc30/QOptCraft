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

import numpy as np


# General number of photons input
def photon_introd(m):
    while True:
        try:
            n = int(input("\nNumber of photons? "))

            if n < 1:
                print("The given value is not valid due to needing at least 1 photon.\n")

            else:
                break

        except ValueError:
            print("The given value is not valid (it is not an integer number).\n")

    # This array is created in order to conserve compatibility with the introduction of an specific photon distribution
    photons = np.zeros(m)

    photons[0] = n

    return photons, n


# Input of an specific distribution of photons within m modes
def photon_introd_one_input(m):
    photons = np.zeros(m, dtype=float)

    while True:
        try:
            for k in range(m):
                n = int(input(f"\nIntroduce the number of photons in the mode {k}: "))

                while n < 0:
                    print("The given value is not valid due to being negative.\n")

                    n = int(input(f"\nIntroduce the number of photons in the mode {k}: "))

                photons[k] = n

            break

        except ValueError:
            print("The given value is not valid (it is not an integer number).\n")

    return photons
