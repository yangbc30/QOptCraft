from setuptools import setup

setup(
	# Se escribe cada cosa seguida de una , al final
	name="QOptCraft",
	version="1.1",
	description="Software package designed for quantum experiments via linear optics instruments. Divided in ten functions (ver 1.1).",
	author="Daniel Gómez Aguado",
	author_email="gomezaguado99@gmail.com",
	#url="",
	license="Copyright 2021 Daniel Gómez Aguado SPDX-License-Identifier: Apache-2.0",
	long_description="QOprCraft navigates through the subalgebras of mainly unitary matrices.\nVia decomposing files into beam splitters and shifters, theoretically desired matrices can be implemented (Selements()), as well as their quantum evolution (read: influence) given a number of photons (StoU()).\nThe reverse operation: giving a desired evolution for finding the original matrix, is also true (SfromU()).\nFor unavailable implementations, the option to obtain its closest, possible evolution matrix is also given via Toponogov().\nSystems with loss are treated by the fifth function QuasiU().\nA matrix generator, a test of some algorithms and every logarithm developed are also included as individual functions.",
	platforms="Developed in Python 3 v3.9.5:0a7dcbd, May 3 2021 17:27:52 for the software Windows-10-10.0.19041-SP0",
	packages=["QOptCraft","QOptCraft.Main_Code","QOptCraft.Phase1_Aux","QOptCraft.Phase2_Aux","QOptCraft.Phase3_Aux","QOptCraft.Phase4_Aux","QOptCraft.Phase5_Aux","QOptCraft.Phase6_Aux"] 
	)
