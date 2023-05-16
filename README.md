# CortExplore_MIMB

Analyse experiments on the cell cortex

1. How to start
	1. Open CortexPaths.py in your favorite code editor.
	2. Make sure to add Code_Python folder (containing CortexPaths.py) to the working directories of the code editor.
	3. Run section "0. Imports" to print your computer name.
	4. Complete section "1.1 Init main directories" by replace all the string starting with '>>>' :
		- '>>>Computer_name' should be your computer name as printed before.
		- '_>>>##' should be your user suffix (ex: _JV for Joseph Vermeil).
		- '>>>User' should be the name of your user session on this computer (check it in C:\Users).
		- If you want to use a backup cloud save of your files (NOT NECESSARY TO USE THE PROGRAM):
			- '>>>ownCloud' should be the location of your main cloud directory.
			- "CloudSaving = '>>>OwnCloud'" should refers to the name of your cloud saving solution (OwnCloud for CNRS agents).	
		NB: This can be done for as many different computers as ones want, and allow to share the exact same programs between different computers without conflicts of directory location.
	5. Run the function CortexPaths.makeDirArchi(). It will automatically create the folder tree used by CortExplore programs. This organisation is described at the end of CortexPaths.py and in the file FileArchitecture.txt.
	6. In the folders, python files and csv files names, every time '_NewUser' is written, replace it by your personnal suffix '_##'. Do not modify '_NewUser2' files: they are there for the next user.
	7. The programs of CortExplore are ready to use! You can start by:
		- Inputing experimental data in 'ExperimentalConditions_##.csv'.
		- Using 'ImagesPreprocessing.py' to crop and import files from a hard drive.
		- Using 'MainDepthoMaker_NewUser.py' to make a depthograph.
		- Using 'MainTracker_NewUser.py' to track the beads in Magnetic Pincher timelapses.
	
For more questions, contact me at joseph.vermeil@espci.fr



***

Joseph Vermeil, 2023
PMMH Laboratories, ESPCI
