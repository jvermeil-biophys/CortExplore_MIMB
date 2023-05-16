# -*- coding: utf-8 -*-
"""
Main_3DTracker.py - Script to call the mainTracker function from SimpleBeadTracker.py.
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

from SimpleBeadTracker import mainTracker


# %% Define paths

dictPaths = {'PathRawDataDir' : 'C://Users//JosephVermeil//Desktop//TestCode_Raw',
             'PathDeptho' : 'C://Users//JosephVermeil//Desktop//TestCode_Deptho//21.04.23_M1_M450_step20_100X_Deptho.tif',
             'PathResultsDir' : 'C://Users//JosephVermeil//Desktop//TestCode_Output',
             }


# %% Define constants

dictConstants = {'bead_type' : 'M450', # 'M450' or 'M270'
                 'bead_diameter' : 4500, # nm
                 'normal_field' : 5, # mT
                 'magnetic_field_correction' : 1.0, # ratio, without unit
                 'multi_images' : 3, # Number of images
                 'multi_image_Z_step' : 500, # nm
                 'multi_image_Z_direction' : 'upward', # Either 'upward' or 'downward'
                 'scale_pixel_per_um' : 15.8, # pixel/µm
                 'optical_index_correction' : 0.85, # ratio, without unit
                 }


# %% Additionnal options

dictOptions = {'redoAllSteps' : False, 
                 'trackAll' : False,
                 'timeLog' : True
                 }


# %% Call mainTracker()

mainTracker(dictPaths, dictConstants, **dictOptions)



