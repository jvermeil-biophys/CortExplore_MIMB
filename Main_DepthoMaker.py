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

# %% Imports

from SimpleBeadTracker import depthoMaker


# %% Define paths

dictPaths = {'PathZStacks' : 'C://Users//JosephVermeil//Desktop//TestCode_ZStacks',
             'PathDeptho' : 'C://Users//JosephVermeil//Desktop//TestCode_Deptho',
             'NameDeptho' : 'Deptho_test.tif',
             }


# %% Define constants

dictConstants = {'bead_type' : 'M450', # 'M450' or 'M270'
                 'bead_diameter' : 4500, # nm
                 'scale_pixel_per_um' : 15.8, # pixel/µm
                 'step' : 20, # nm
                 }


# %% Call depthoMaker()

depthoMaker(dictPaths, dictConstants)

