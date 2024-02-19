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
# %% How to use this script

"""
1. Run the cell "1. Imports". If an error is returned, a common fix is to define 
    the folder containing this script as your current working directory.
2. In the cell "2. Define paths", fill the dictionnary with the corrects paths. 
    Then run this cell.
3. In the cell "3. Define constants", indicate the relevant values for 
    the parameters that will be used in the program. Then run this cell.
4. In the cell "4. Additionnal options", adjust the values of the settings accordingly.
    Then run this cell.
5. Finally, run the cell "5. Call mainTracker()" without modifying it.
"""

# %% 1. Imports

from SimpleBeadTracker import mainTracker


# %% 2. Define paths

dictPaths = {'PathRawDataDir' : './/Example_Data_2024//01_Timelapses',
             'PathDeptho'     : './/Example_Data_2024//03_Depthograph//23-09-11_Deptho.tif', # 
             'PathResultsDir' : './/Example_Data_2024//04_Results',
             }


# =============================================================================
# DESCRIPTION
# 'PathRawDataDir' : the path to the folder containing your raw data, meaning 
#                    your timelapses in .tif format, with the associated .txt 
#                    files ("_Results.txt" and "Timepoints.txt").
# 'PathDeptho'     : the path to the depthograph in .tif format.
# 'PathResultsDir' : the path to the folder where you want to save the results.
# 
# ATTENTION ! The default values are the one you need to analyse the example dataset.
# =============================================================================

# %% 3. Define constants

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

# =============================================================================
# DESCRIPTION
# bead type                 : text
#                             Identify the bead type. Default is M450.
# 
# bead diameter             : int
#                             Identify the bead diameter in nm. Need to be determined 
#                             from a calibration for each new batch of beads.
# 
# normal field              : float
#                             Uniform magnetic field applied to the chamber during 
#                             the experiment, in mT. It is the *target* field. 
#                             The actual field applied one might differ: see next parameter.
# 
# magnetic field correction : float
#                             Corrective multiplicative coefficient to 'normal field' 
#                             to take into account the actual magnetic field of the experiment. 
#                             Example: one does an experiment with a "target" constant field at 5 mT. 
#                             At the end of the experiment, one measures with a gaussmeter 
#                             than the field was actually 5.1 mT. The 'magnetic field correction' 
#                             will be 5.1 / 5 = 1.02.
# 
# multi images              : int
#                             Number of Z-planes acquired per timepoint. Acquiring 
#                             several Z-planes per timepoint increase the precision of Z-detection.
#                             Default is 3.
# 
# multi image Z step        : int
#                             Step in Z between each Z-planes if multi-image > 1, in nm.
#                             Default is 500.
# 
# multi image Z direction   : text
#                             Direction of the scan when acquiring several Z-planes 
#                             per timepoint. Needs to be 'upward' or 'downward'.
#                             Default is 'upward'.
# 
# scale pixel per um        : float
#                             Scale of the objective in pixel per micron. 
#                             Proceeding to a manual calibration when using 
#                             a new microscope is very strongly recommended.
# 
# optical index correction  : float
#                             Ratio of the optical index of the cell medium over 
#                             the index of the immersion liquid. Crucial for Z-distances 
#                             computation. When using an oil-objective and typical DMEM, 
#                             optical index correction = 1.33/1.52 = 0.875. When using 
#                             an air-objective and typical DMEM, optical index correction = 1.33/1.00 = 1.33. 
# =============================================================================

# %% 4. Additionnal options

dictOptions = {'redoAllSteps' : False, 
               'timepoints' : True
              }

# =============================================================================
# DESCRIPTION
# redoAllSteps              : bool, default is False.
#                             If False, the programm will use intermediate outputs 
#                             to skip steps that would have already been performed, thus saving time
#                             (for instance if you reanalyze a given movie for the second time).
#                             If True, the programm will ignore these intermediate outputs and perform the full analysis.
# 
# timepoints                : bool, default is True.
#                             If True, the programm will expect a "_Timepoints.txt" text file,
#                             containing a single column of time points in ms, corresponding to every frame in the image file. 
#                             This time column will be used to generate the results.
#                             If False, the program will bypass this by generating a mock time column, 
#                             with integers from 1 to N, N being the number of frames.
# =============================================================================


# %% 5. Call mainTracker()

mainTracker(dictPaths, dictConstants, **dictOptions)



