==================
   DESCRIPTION
==================

Author: Daniel Gómez Aguado
e-mail: gomezaguado99@gmail.com

QOprCraft is a software package or library developed in Python 3, designed for quantum experiments building via linear optics instruments. 
Divided in ten functions (ver 1.1). It navigates through the subalgebras of mainly unitary matrices.
1) Via decomposing files into beam splitters and shifters, theoretically desired matrices can be implemented (Selements()), 
2) as well as their quantum evolution (read: influence) given a number of photons (StoU()).
3) The reverse operation: giving a desired evolution for finding the original matrix, is also true (SfromU()).
4) For unavailable implementations, the option to obtain its closest, possible evolution matrix is also given via Toponogov().
5) Systems with loss are treated by the fifth function QuasiU().
6) Measurement via computing the Schmidt rank of Fock states vectors after passing a quantum through a given linear optics circuit (StateSchmidt()).
A matrix generator, a test of some algorithms and every logarithm developed are also included as individual functions.

"QOptCraft()" is the main function, making full use of all the algorithms available. 
Its standalone subfunctions or blocks (read user guide) can be deployed on their own as well.
Use the module parameter (1-10) for picking which subfunction to use: 
> Selements (module=1)
> StoU (module=2)
> SfromU (module=3)
> Toponogov (module=4)
> QuasiU (module=5)
> StateSchmidt (module=6)
> iHStoiHU (module=7).
> Use the choice parameter for subsubfunctions in QOCGen (module=8, choice=1-7), QOCTest (module=9, choice=1-3) or QOCLog (module=10, choice=1-5).
More info on the remaining parameters by reading QOptCraft's user guide.



==========================================================
  LICENSE NOTICE (SOFTWARE PACKAGE QOptCraft-1.1.tar.gz)
==========================================================

   QOptCraft

   Copyright 2021 Daniel Gómez Aguado

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.



The user guide (QOptCraft_V1.1_user_guide.pdf) is under no license.