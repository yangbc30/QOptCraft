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

# ---------------------------------------------------------------------------------------------------------------------------
# 													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------


# ----------MATHEMATICAL FUNCTIONS/MATRIX OPERATIONS:----------

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy


# ----------SYSTEM:----------

import sys


# ---------------------------------------------------------------------------------------------------------------------------
# 													INPUT CONTROL
# ---------------------------------------------------------------------------------------------------------------------------


def decimal_precision():
    while True:
        try:
            precision = int(input("\nHow many decimals will be requested?: "))

            if precision < 0:
                print("The given value is not valid.\n")

            else:
                return precision

        except ValueError:
            print("The given value is not valid.\n")


def input_control(
    module=1,
    file_input=True,
    M_input=False,
    file_output=True,
    filename=False,
    txt=False,
    acc_d=3,
    newfile=True,
):
    if (
        file_input is False
        and isinstance(M_input, numpy.ndarray) is False
        and (module != 1 and module != 5)
    ):
        if isinstance(M_input, numpy.ndarray) is False:
            print("\nWARNING: invalid matrix input.")

        while True:
            try:
                check = int(input("\nOpen matrix from a file?:\n1) Yes \n2) No\n"))

                if check == 2:
                    print("\nQOptCraft will end.")

                    sys.exit()

                elif check == 1:
                    newfile = False

                    file_input = True

                    if filename is not False:
                        # Name adapted to str
                        filename = str(filename)

                    else:
                        print("\nWARNING: a new/loaded filename is required.")

                    while filename is False:
                        try:
                            filename = input(
                                "Write the name of the file (without .txt extension): "
                            )

                        except ValueError:
                            print("The given value is not valid.\n")

                    break

            except ValueError:
                print("The given value is not valid.\n")

    if filename is not False:
        # Name adapted to str
        filename = str(filename)

    elif file_output is True or file_input is True:
        print("\nWARNING: a new/loaded filename is required.")

        while filename is False:
            try:
                filename = input("Write the name of the file (without .txt extension): ")

            except ValueError:
                print("The given value is not valid.\n")

    if txt is True:
        if acc_d < 0 or type(acc_d) is not int:
            if acc_d < 0:
                print(
                    "\nWARNING: interactive mode requires an output decimal precision of 0 or higher."
                )

            elif type(acc_d) is not int:
                print(
                    "\nWARNING: invalid precision input (needs to be int and equal or higher than 0)."
                )

            # This variable corresponds to the decimal precision. It applies to the results presented onscreen
            # In the .txt files, results contain all possible decimals
            acc_d = decimal_precision()

    return file_input, filename, newfile, acc_d


def input_control_ints(var, var_name, cond):
    if var is False or var < cond or type(var) is not int:
        if var is False:
            print(f"\nWARNING: when creating a new file, a variable {var_name} must be given.")

        elif var < cond or type(var) is not int:
            print(
                f"\nWARNING: invalid {var_name} input (needs to be int and equal or higher than {cond})."
            )

        while True:
            try:
                var = int(input(f"\nValue of {var_name}: "))

                if var >= cond:
                    break

                print(
                    f"\nWARNING: invalid {var_name} input (needs to be int and equal or higher than {cond})."
                )

            except ValueError:
                print("The given value is not valid.\n")

    return var


def input_control_intsDim(var, var_name, cond):
    if var is False or var < cond or type(var) is not int:
        if var is False:
            print(
                f"\nWARNING: when creating a new file, a variable {var_name} (size {var_name} of the {var_name} x {var_name} matrix) must be given."
            )

        elif var < cond or type(var) is not int:
            print(
                f"\nWARNING: invalid {var_name} input (needs to be int and equal or higher than {cond})."
            )

        while True:
            try:
                var = int(
                    input(
                        f"\nValue of {var_name} (size {var_name} of the {var_name} x {var_name} matrix): "
                    )
                )

                if var >= cond:
                    break

                print(
                    f"\nWARNING: invalid {var_name} input (needs to be int and equal or higher than {cond})."
                )

            except ValueError:
                print("The given value is not valid.\n")

    return var


def input_control_floats(var, var_name, cond):
    if var is False or var < cond or type(var) is not float:
        if var is False:
            print(f"\nWARNING: when creating a new file, a variable {var_name} must be given.")

        elif var < cond or type(var) is not float:
            print(
                f"\nWARNING: invalid {var_name} input (needs to be int and equal or higher than {cond})."
            )

        while True:
            try:
                var = float(input(f"\nValue of {var_name}: "))

                if var >= cond:
                    break

                print(
                    f"\nWARNING: invalid {var_name} input (needs to be int and equal or higher than {cond})."
                )

            except ValueError:
                print("The given value is not valid.\n")

    return var
