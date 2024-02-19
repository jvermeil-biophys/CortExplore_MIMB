# -*- coding: utf-8 -*-
"""
Main_DepthoMaker.py - Script to use the depthoMaker function from SimpleBeadTracker.py
Joseph Vermeil, 2023

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

# %% How to use this script

"""
1. Run the cell "1. Imports". If an error is returned, a common fix is to define 
    the folder containing this script as your current working directory.
2. In the cell "2. Define paths", fill the dictionnary with the corrects paths. 
    Then run this cell.
3. In the cell "3. Define constants", indicate the relevant values for 
    the parameters that will be used in the program. Then run this cell.
4. Finally, run the cell "4. Call depthoMaker()" without modifying it.
"""

# %% 1. Imports

from SimpleBeadTracker import depthoMaker


# %% 2. Define paths

dictPaths = {'PathZStacks' : './/Example_Data_2024//02_ZScans',
             'PathDeptho'  : './/Example_Data_2024//03_Depthograph', # {os.getcwd()}
             'NameDeptho'  : '23-09-11_Deptho.tif',
             }

# =============================================================================
# DESCRIPTION
# 'PathZStacks' : the path to the folder containing your raw data, meaning your Z-stacks 
#                 in .tif format, with the associated .txt files ("_Results.txt").
# 'PathDeptho'  : the path to the folder where you want to save the depthograph data.
# 'NameDeptho'  : the name you want to give to the depthograph (need to end with '.tif').
# 
# ATTENTION ! The default values are the one you need to analyse the example dataset.
# =============================================================================

# %% 3. Define constants

dictConstants = {'bead type' : 'M450', # 'M450' or 'M270'
                 'scale pixel per um' : 15.8, # pixel/µm
                 'step' : 20, # nm
                 }

# =============================================================================
# DESCRIPTION
# bead type                 : text
#                             Identify the bead type. Default is M450.
# 
# scale pixel per um        : float
#                             Scale of the objective in pixel per micron. 
#                             Proceeding to a manual calibration when using 
#                             a new microscope is very strongly recommended.
# 
# step                      : int
#                             The step between each frame of the source Z-scans, in nm.
#                             Our typical Z-scans are made of 401 frames with 20 nm steps.
# =============================================================================

# %% 4. Call depthoMaker()

depthoMaker(dictPaths, dictConstants)

