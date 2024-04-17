# -*- coding: utf-8 -*-
"""
SimpleBeadTracker.py - contains the classes to perform bead tracking in a movie
(see the function mainTracker and the Tracker classes), and to make a Depthograph
(see the function depthoMaker and the Depthograph classes).
Joseph Vermeil, Anumita Jawahar, 2023

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

# %% (0) Imports and settings

# 1. Imports
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import os
import time
import pyautogui

from scipy import interpolate
from matplotlib import ticker
from skimage import io, transform
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment


# 2. Local Imports
import GraphicStyles as gs
import UtilityFunctions as ufun


# 3. Advanced settings

K_CORRMAG = 1.0


# %% (1) Tracker classes


# %%%% PincherTimeLapse

class PincherTimeLapse:
    """
    This class is initialised for each new .tif file analysed.

    It requires the following inputs :
    > I : the timelapse analysed.
    > cellID : the id of the cell currently analysed.
    > manipDict : the line of the experimental data table that concerns the current experiment.
    > NB : the number of beads of interest that will be tracked.

    It contains:
    * data about the 3D image I (dimensions = time, height, width),
    * a list of Frame objects listFrames, 1 per frame in the timelapse,
    * a list of Trajectory objects listTrajectories, 1 per bead of interest (Boi) in the timelapse,
    * a dictionnary dictLog, saving the status_frame of each frame (see below)
                             and all of the user inputs (points and clicks) during the tracking,
    * a pandas DataFrame detectBeadsResult, that contains the raw output of the bead tracking,
    * metadata about the experiment (cellID, expType, loopStruct, Nuplet).

    When a PincherTimeLapse is initialised, most of these variables are initialised to zero values.
    In order to compute the different fields, the following methods should be called in this order:
    - PTL.determineFramesStatus() : fill the status_frame and status_nUp column of the dictLog.
                                    in the status_frame field: -1 means excluded ; 0 means singlet ; >0 means *position in* the n-uplet
                                    in the status_nUp field: -1 means excluded ; 0 means singlet ; >0 means *number of* the n-uplet
    - PTL.saveMetaData() : Save the computed threshold along with a few other data.
    - PTL.makeFramesList() : Initialize the frame list.
    - PTL.detectBeads() : Detect all the beads or load their positions from a pre-existing '_Results.txt' file.
    - PTL.buildTrajectories() : Do the tracking of the beads of interest, with the user help, or load pre-existing trajectories.
    [In the meantime, Z computations and neighbours detections are performed on the Trajectory objects]
    - PTL.computeForces() : when the Trajectory objects are complete (with Z and neighbours), compute the forces.
                            Include recent corrections to the formula [October 2021].
    """

    def __init__(self, I, cellID, dictConstants):
        # 1. Infos about the 3D image. The shape of the 3D image should be the following: T, Y, X !
        nS, ny, nx = I.shape[0], I.shape[1], I.shape[2]
        self.I = I
        self.nx = nx
        self.ny = ny
        self.nS = nS

        # 2. Infos about the experimental conditions, mainly from the DataFrame 'manipDict'.
        self.NB = 2 # The number of beads of interest ! 2 for a normal experiment
        self.cellID = cellID
        self.expType = 'constant field'
        self.scale = dictConstants['scale_pixel_per_um']
        self.OptCorrFactor = dictConstants['optical_index_correction']
        self.MagCorrFactor = dictConstants['magnetic_field_correction']
        self.Nuplet = dictConstants['multi_images']
        self.Zstep = dictConstants['multi_image_Z_step']
        self.Zdirection = dictConstants['multi_image_Z_direction']
        self.MagField = dictConstants['normal_field']

        self.beadType = dictConstants['bead_type']
        self.beadDiameter = int(dictConstants['bead_diameter'])
        # self.microscope = manipDict['microscope software']

        self.loopStructRaw = np.nan
        self.ignore_loopStructure = (pd.isnull(self.loopStructRaw))
        
        # 3. Field that are just initialized for now and will be filled by calling different methods.
        self.listFrames = []
        self.listTrajectories = []

        self.dictLog = {'Slice' : np.array([i+1 for i in range(nS)]),
                        'status_frame' : np.zeros(nS, dtype = float),  # in the status_frame field: -1 means excluded ; 0 means singlet ; >0 means position in the n-uplet
                        'status_nUp' : np.zeros(nS, dtype = int), # in the status_nUp field: -1 means excluded ; 0 means singlet ; >0 means number of the n-uplet
                        'UI' : np.zeros(nS, dtype = bool),
                        'UILog' : np.array(['' for i in range(nS)], dtype = '<U16'),
                        'UIxy' : np.zeros((nS,self.NB,2), dtype = int)}

        self.detectBeadsResult = pd.DataFrame({'Area' : [],
                                               'StdDev' : [],
                                               'XM' : [],
                                               'YM' : [],
                                               'Slice' : []})

        
        self.modeNoUIactivated = False
        # End of the initialization !
        
        
        
    def determineFramesStatus(self):
        """
        Fill the status_frame and status_nUp column of the dictLog
        > in the status_frame field: -1 means excluded ; 0 means singlet ; 10 > x > 0 means *position in* the n-uplet.
        > in the status_nUp field: -1 means excluded ; 0 means singlet ; >0 means *number of* the n-uplet.
        """
        
        if self.ignore_loopStructure:
            if self.Nuplet == 1:
                self.dictLog['status_frame'] = np.zeros(self.nS, dtype = int)
                self.dictLog['status_nUp']
            elif self.Nuplet > 1:
                A = np.arange(self.nS, dtype = int)
                self.dictLog['status_frame'] = 1 + A%self.Nuplet
                self.dictLog['status_nUp'] = 1 + A//self.Nuplet
            
        else:
            expType = self.expType
            # Do sth according to the expType.
            
            if expType == 'constant field':
                if self.Nuplet == 1:
                    self.dictLog['status_frame'] = np.zeros(self.nS, dtype = int)
                    self.dictLog['status_nUp']
                elif self.Nuplet > 1:
                    A = np.arange(self.nS, dtype = int)
                    self.dictLog['status_frame'] = 1 + A%self.Nuplet
                    self.dictLog['status_nUp'] = 1 + A//self.Nuplet
                    
            elif expType == 'other experiment type':
                pass # Complete if needed
       
                
    def saveLog(self, display = 1, save = False, path = ''):
        """
        Save the dictLog so that next time it can be directly reloaded to save time.
        """
        dL = {}
        dL['Slice'], dL['status_frame'], dL['status_nUp'] = \
            self.dictLog['Slice'], self.dictLog['status_frame'], self.dictLog['status_nUp']

        dL['UI'], dL['UILog'] = \
            self.dictLog['UI'], self.dictLog['UILog']
        for i in range(self.NB):
            dL['UIx'+str(i+1)] = self.dictLog['UIxy'][:,i,0]
            dL['UIy'+str(i+1)] = self.dictLog['UIxy'][:,i,1]
        dfLog = pd.DataFrame(dL)
        if save:
            dfLog.to_csv(path, sep='\t')

        if display == 1:
            print('\n\n* Initialized Log Table:\n')
            print(dfLog)
        if display == 2:
            print('\n\n* Filled Log Table:\n')
            print(dfLog[dfLog['UI']])

    def importLog(self, path):
        """
        Import the dictLog.
        """
        dfLog = pd.read_csv(path, sep='\t')
        dL = dfLog.to_dict()
        self.dictLog['Slice'], self.dictLog['status_frame'], self.dictLog['status_nUp'] = \
            dfLog['Slice'].values, dfLog['status_frame'].values, dfLog['status_nUp'].values
        self.dictLog['UI'], self.dictLog['UILog'] = \
            dfLog['UI'].values, dfLog['UILog'].values
        for i in range(self.NB):
            xkey, ykey = 'UIx'+str(i+1), 'UIy'+str(i+1)
            self.dictLog['UIxy'][:,i,0] = dfLog[xkey].values
            self.dictLog['UIxy'][:,i,1] = dfLog[ykey].values        
            
            
    def makeFramesList(self):
        """
        Initialize the Frame objects and add them to the PTL.listFrames list.
        """
        for i in range(self.nS):
            status_frame = self.dictLog['status_frame'][i]
            status_nUp = self.dictLog['status_nUp'][i]
            # The Nup field of a slice is = to self.Nuplet if the status_frame indicates that the frmae is part of a multi image n-uplet
            # Otherwise the image is "alone", like in a compression, and therefore Nup = 1
            Nup = (self.Nuplet * (status_nUp > 0))  +  (1 * (status_nUp <= 0))
            if self.dictLog['status_frame'][i] >= 0:
                self.listFrames.append(Frame(self.I[i], i, self.NB, Nup, status_frame, status_nUp, self.scale))
                

    def detectBeads(self):
        """
        If no '_Results.txt' file has been previously imported, ask each Frame
        object in the listFrames to run its Frame.detectBeads() method.
        Then concatenate the small 'Frame.resDf' to the big 'PTL.detectBeadsResult' DataFrame,
        so that in the end you'll get a DataFrame that has exactly the shape of a '_Results.txt' file made from IJ.
        *
        If a '_Results.txt' file has been previously imported, just assign to each Frame
        object in the listFrames the relevant resDf DataFrame
        (each resDf is, as said earlier, just a fragment of the PTL.detectBeadsResult).
        """
        for frame in self.listFrames: #[:3]:
            plot = 0
            # if not resFileImported:
            #     frame.detectBeads(plot)
            #     self.detectBeadsResult = pd.concat([self.detectBeadsResult, frame.resDf])

            # else:
            resDf = self.detectBeadsResult.loc[self.detectBeadsResult['Slice'] == frame.iS+1]
            frame.resDf = resDf

            frame.makeListBeads()

    # def saveBeadsDetectResult(self, path):
    #     """
    #     Save the 'PTL.detectBeadsResult' DataFrame.
    #     """
    #     self.detectBeadsResult.to_csv(path, sep='\t', index = False)

    def importBeadsDetectResult(self, path=''):
        """
        Import the 'PTL.detectBeadsResult' DataFrame.
        """
        df = pd.read_csv(path, sep='\t')
        for c in df.columns:
            if 'Unnamed' in c:
                df.drop([c], axis = 1, inplace=True)
                
        self.detectBeadsResult = df

        
    
    
    def buildTrajectories(self, trackAll = False):
        """
        The main tracking function.
        *
        Note about the naming conventions here:
        - 'iF': index in the list of Frames ;
        - 'iB': index in a list of Beads or a list of Trajectories ;
        - 'iS': index of the slice in the image I (but here python starts with 0 and IJ starts with 1);
        - 'Boi' refers to the 'Beads of interest', ie the beads that are being tracked.
        """
        
        #### 1. Initialize the BoI position in the first image where they can be detect, thanks to user input.
        init_iF = 0
        init_ok = False
        while not init_ok:
            init_iS = self.listFrames[init_iF].iS

            if not self.dictLog['UI'][init_iS]: # Nothing in the log yet
                self.listFrames[init_iF].show()
                #### Windows specific
#                 mngr = plt.get_current_fig_manager()
#                 mngr.window.setGeometry(720, 50, 1175, 1000)
                QA = pyautogui.confirm(
                    text='Can you point the beads of interest\nin the image ' + str(init_iS + 1) + '?',
                    title='Initialise tracker',
                    buttons=['Yes', 'Next Frame', 'Quit'])
                if QA == 'Yes':
                    init_ok = True
                    ui = plt.ginput(self.NB, timeout=0)
                    uiXY = ufun.ui2array(ui)
                    self.dictLog['UI'][init_iS] = True
                    self.dictLog['UILog'][init_iS] = 'init_' + QA
                    self.dictLog['UIxy'][init_iS] = uiXY
                elif QA == 'Next Frame':
                    self.dictLog['UI'][init_iS] = True
                    self.dictLog['UILog'][init_iS] = 'init_' + QA
                    init_iF += 1
                else:
                    fig = plt.gcf()
                    plt.close(fig)
                    return('Bug')

                fig = plt.gcf()
                plt.close(fig)

            else: # Action to do already in the log
                QA = self.dictLog['UILog'][init_iS]
                if QA == 'init_Yes':
                    init_ok = True
                    uiXY = self.dictLog['UIxy'][init_iS]
                elif QA == 'init_Next Frame':
                    init_iF += 1
                else:
                    print('Strange event in the tracking init')

        init_BXY = self.listFrames[init_iF].beadsXYarray()
        M = ufun.compute_cost_matrix(uiXY,init_BXY)
        row_ind, col_ind = linear_sum_assignment(M) # row_ind -> clicks / col_ind -> listBeads
        
        # Sort the beads by growing X coordinates on the first image,
        # So that iB = 0 has a X inferior to iB = 1, etc.
        sortM = np.array([[init_BXY[col_ind[i],0], col_ind[i]] for i in range(len(col_ind))])
        sortM = sortM[sortM[:, 0].argsort()]
        
        # Initialise position of the beads
        init_iBoi = sortM[:, 1].astype(int)
        # init_BoiXY = sortM[:, 0]
        init_BoiXY = np.array([init_BXY[init_iBoi[i]] for i in range(len(init_iBoi))])
        
        
        #### 2. Creation of the Trajectory objects
        for iB in range(self.NB):
            self.listTrajectories.append(Trajectory(self.I, self.cellID, self.listFrames, self.scale, self.Zstep, iB))

            self.listTrajectories[iB].dict['Bead'].append(self.listFrames[init_iF].listBeads[init_iBoi[iB]])
            self.listTrajectories[iB].dict['iF'].append(init_iF)
            self.listTrajectories[iB].dict['iS'].append(self.listFrames[init_iF].iS)
            self.listTrajectories[iB].dict['iB_inFrame'].append(init_iBoi[iB])
            self.listTrajectories[iB].dict['X'].append(init_BoiXY[iB][0])
            self.listTrajectories[iB].dict['Y'].append(init_BoiXY[iB][1])
            self.listTrajectories[iB].dict['StdDev'].append(self.listFrames[init_iF].beadsStdDevarray()[init_iBoi[iB]])
            self.listTrajectories[iB].dict['status_frame'].append(self.listFrames[init_iF].status_frame)
            self.listTrajectories[iB].dict['status_nUp'].append(self.listFrames[init_iF].status_nUp)
            
            self.listTrajectories[iB].dict['idxAnalysis'].append(0)
            

        #### 3. Start the tracking
        previous_iF = init_iF
        previous_iBoi = init_iBoi
        previous_BXY = init_BXY
        previous_BoiXY = init_BoiXY
        
        
        for iF in range(init_iF+1, len(self.listFrames)):
            validFrame = True
            askUI = False
            
            #### 3.1 Check the number of detected objects
            if self.listFrames[iF].NBdetected < self.NB: # -> Next frame
                validFrame = False
                continue
            
            #### 3.2 Try an automatic tracking
            if not trackAll:
                trackXY = previous_BoiXY
                previous_iBoi = [i for i in range(self.NB)]
            elif trackAll:
                trackXY = previous_BXY
                
            BXY = self.listFrames[iF].beadsXYarray()
            M = ufun.compute_cost_matrix(trackXY,BXY)
            row_ind, col_ind = linear_sum_assignment(M)
            costs = np.array([M[row_ind[iB], col_ind[iB]] for iB in range(len(row_ind))])
            foundBoi = []
            for iBoi in previous_iBoi:
                searchBoi = np.flatnonzero(row_ind == iBoi)
                if len(searchBoi) == 1:
                    foundBoi.append(searchBoi[0])
                                   
            
            #### 3.3 Assess if asking user input is necessary
            
            highCost = ((np.max(costs)**0.5) * (1/self.scale) > 0.5)
            # True if the distance travelled by one of the BoI is greater than 0.5 um
            
            allBoiFound = (len(foundBoi) == self.NB)
            # False if one of the beads of interest have not been detected
            
            if highCost or not allBoiFound:
                askUI = True
                
            #### 3.4 If not, automatically assign the positions of the next beads

            if not askUI:
                try:
                    iBoi = [col_ind[iB] for iB in foundBoi]
                    BoiXY = np.array([BXY[iB] for iB in iBoi])
                    
                except:
                    askUI = True
                    print('Error for ' + str(iF))
                    print('M')
                    print(M)
                    print('row_ind, col_ind')
                    print(row_ind, col_ind)
                    print('previous_iBoi')
                    print(previous_iBoi)
                    print('costs')
                    print(costs)
                    

            #### 3.5 If one of the previous steps failed, ask for user input
            if askUI:        
                iS = self.listFrames[iF].iS
                
                #### 3.5.1: Case when the UI has been previously saved in the dictLog.
                # Then just import the previous answer from the dictLog
                if self.dictLog['UI'][iS]:
                    QA = self.dictLog['UILog'][iS]
                    if QA == 'Yes':
                        
                        uiXY = self.dictLog['UIxy'][iS]
                    elif QA == 'No' or QA == 'No to all':
                        validFrame = False
                        #fig = plt.gcf()
                        #plt.close(fig)
                
                
                #### 3.5.2: Case when the UI has NOT been previously saved in the dictLog
                # Then ask for UI ; and save it in the dictLog
                elif not self.dictLog['UI'][iS]:
                    if self.modeNoUIactivated == False:
                        # Display the image, plot beads positions and current trajectories & ask the question
                        self.listFrames[iF].show()
                        for iB in range(self.NB):
                            T = self.listTrajectories[iB]
                            ax = plt.gca()
                            T.plot(ax, iB)
                        
                        #### Windows specific
                        # mngr = plt.get_current_fig_manager()
                        # mngr.window.setGeometry(720, 50, 1175, 1000)
                        QA = pyautogui.confirm(
                            text='Can you point the beads of interest\nin the image ' + str(iS + 1) + '?',
                            title='', 
                            buttons=['No', 'Yes', 'Abort!', 'No to all'])
                        
                        # According to the question's answer:
                        if QA == 'Yes':
                            ui = plt.ginput(self.NB, timeout=0)
                            uiXY = ufun.ui2array(ui)
                            self.dictLog['UI'][iS] = True
                            self.dictLog['UILog'][iS] = QA
                            self.dictLog['UIxy'][iS] = uiXY
                        elif QA == 'No':
                            validFrame = False
                            self.dictLog['UI'][iS] = True
                            self.dictLog['UILog'][iS] = QA
                        elif QA == 'Abort!':
                            validFrame = False
                            fig = plt.gcf()
                            plt.close(fig)
                            return('Bug')
                        elif QA == 'No to all':
                            validFrame = False
                            self.modeNoUIactivated = True
                            self.dictLog['UI'][iS] = True
                            self.dictLog['UILog'][iS] = QA
                        fig = plt.gcf()
                        plt.close(fig)
                        
                    elif self.modeNoUIactivated == True:
                    # This mode is in case you don't want to keep clicking 'No' for hours when
                    # you know for a fact that there is nothing else you can do with this TimeLapse.
                        iS = self.listFrames[iF].iS
                        QA = 'No'
                        validFrame = False
                        self.dictLog['UI'][iS] = True
                        self.dictLog['UILog'][iS] = QA
                
                #### 3.5.3: Outcome of the user input case
                if not validFrame: # -> Next Frame
                    continue
            
                else:
                    # Double matching here
                    # First you match the user's click positions with the bead positions detected on frame iF
                    # You know then that you have identified the NB Beads of interest.
                    # Then another matching between these two new UIfound_BoiXY and the previous_BoiXY
                    # to be sure to attribute each position to the good trajectory !
                    
                    # First matching
                    M = ufun.compute_cost_matrix(uiXY,BXY)
                    row_ind, col_ind = linear_sum_assignment(M)
                    UIfound_BoiXY = np.array([BXY[iB] for iB in col_ind])
                    
                    # Second matching
                    M2 = ufun.compute_cost_matrix(previous_BoiXY, UIfound_BoiXY)
                    row_ind2, col_ind2 = linear_sum_assignment(M2)

                    
                    iBoi = [col_ind[i] for i in col_ind2]
                    BoiXY = np.array([BXY[iB] for iB in iBoi])

                    
            #### 3.6 Create the 'idxAnalysis' field
            if self.expType == 'constant field':
                idxAnalysis = 0
            
            
            #### 3.7 Append the different lists of listTrajectories[iB].dict
            for iB in range(self.NB):
                self.listTrajectories[iB].dict['Bead'].append(self.listFrames[iF].listBeads[iBoi[iB]])
                self.listTrajectories[iB].dict['iF'].append(iF)
                self.listTrajectories[iB].dict['iS'].append(self.listFrames[iF].iS)
                self.listTrajectories[iB].dict['iB_inFrame'].append(iBoi[iB])
                self.listTrajectories[iB].dict['X'].append(BoiXY[iB][0])
                self.listTrajectories[iB].dict['Y'].append(BoiXY[iB][1])
                self.listTrajectories[iB].dict['StdDev'].append(self.listFrames[iF].beadsStdDevarray()[iBoi[iB]])
                self.listTrajectories[iB].dict['status_frame'].append(self.listFrames[iF].status_frame)
                self.listTrajectories[iB].dict['status_nUp'].append(self.listFrames[iF].status_nUp)
                self.listTrajectories[iB].dict['idxAnalysis'].append(idxAnalysis)
            

            #### 3.8 Initialize the next passage in the loop
            previous_iF = iF
            previous_iBoi = iBoi
            previous_BXY = BXY
            previous_BoiXY = BoiXY
            
            
            
            #### 3.9 End of the loop
            
                
        for iB in range(self.NB):
            for k in self.listTrajectories[iB].dict.keys():
                self.listTrajectories[iB].dict[k] = np.array(self.listTrajectories[iB].dict[k])
                
        
        #### 4.1 Refine the trajectories
        
        # The next paragraph was removed in SimpleBeadTracker but might become useful again.
        
        # nT = len(self.listTrajectories[0].dict['Bead'])
        # for iB in range(self.NB):
        #     self.listTrajectories[iB].dict['Zr'] = np.zeros(nT)
        #     self.listTrajectories[iB].nT = nT
        #     iField = []
        #     for i in range(nT):
        #         iF = self.listTrajectories[iB].dict['iF'][i]
        #         SField = iF
        #         iField.append(SField)
        #     self.listTrajectories[iB].dict['iField'] = iField
        
        # The simpler version.
        
        nT = len(self.listTrajectories[0].dict['Bead'])
        for iB in range(self.NB):
            self.listTrajectories[iB].dict['Zr'] = np.zeros(nT)
            self.listTrajectories[iB].nT = nT
            self.listTrajectories[iB].dict['iField'] = self.listTrajectories[iB].dict['iF']

            
        #### 4.2 Find the image with the best std within each n-uplet
            
        bestStd = self.findBestStd()
        for i in range(self.NB):
            self.listTrajectories[i].dict['bestStd'] = bestStd
            
            
    def plot_computed_trajectories(self):
        lastF = self.listFrames[-1]
        fig, ax = plt.subplots(1,1, figsize=(5, 5))
        ax.imshow(lastF.F, cmap = 'gray', vmin = 0, vmax = 3000)
        # if len(lastF.listBeads) > 0:
        #     for B in lastF.listBeads:
        #         ax.plot([B.x], [B.y], c='darkred', marker='P', markersize = 15)       
        for iB in range(self.NB):
            T = self.listTrajectories[iB]
            colors = gs.colorList10
            c = colors[iB]
            ax.plot(T.dict['X'], T.dict['Y'], color=c, lw=0.75)
            if iB == 0:
                ax.plot(T.dict['X'][0], T.dict['Y'][0], lw=0, 
                        color='darkturquoise', marker = 'o', markersize = 5, zorder=4, label='initial')
                ax.plot(T.dict['X'][-1], T.dict['Y'][-1], lw=0,
                        color='darkred', marker = 'o', markersize = 5, zorder=5, label='final')
            else:
                ax.plot(T.dict['X'][0], T.dict['Y'][0], lw=0, 
                        color='darkturquoise', marker = 'o', markersize = 5, zorder=4)
                ax.plot(T.dict['X'][-1], T.dict['Y'][-1], lw=0,
                        color='darkred', marker = 'o', markersize = 5, zorder=5)
        ax.set_xlim([125, 315])   
        ax.set_ylim([460, 270])
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.legend(loc='lower right', fontsize = 9)
        
        plt.show()
        self.plot_summary = (fig, ax)
            


    def importTrajectories(self, path, iB):
        """
        """
        self.listTrajectories.append(Trajectory(self.I, self.cellID, self.listFrames, self.scale, self.Zstep, iB))
        traj_df = pd.read_csv(path, sep = '\t')
        cols = traj_df.columns.values
        cols_to_remove = []
        for c in cols:
            if 'Unnamed' in c:
                cols_to_remove.append(c)
        traj_df = traj_df.drop(columns = cols_to_remove)
        self.listTrajectories[-1].dict = traj_df.to_dict(orient = 'list')
        for i in range(len(self.listTrajectories[-1].dict['iF'])):
            iBoi =  self.listTrajectories[-1].dict['iB_inFrame'][i]
            iF =  self.listTrajectories[-1].dict['iF'][i]
            self.listTrajectories[-1].dict['Bead'][i] = self.listFrames[iF].listBeads[iBoi]



    def findBestStd(self):
        """
        Simpler and better than findBestStd_V0 using the status_nUp column of the dictLog.
        ---
        For each frame of the timelapse that belongs to a N-uplet, I want to reconsititute this N-uplet
        (meaning the list of 'Nup' consecutive images numbered from 1 to Nup,
        minus the images eventually with no beads detected).
        Then for each N-uplet of images, i want to find the max standard deviation
        and report its position because it's for the max std that the X and Y detection is the most precise.
        ---
        This is very easy thanks to the 'status_nUp', because it contains a different number for each N-Uplet.
        """

        Nup = self.Nuplet
        nT = self.listTrajectories[0].nT
        status_nUp = self.listTrajectories[0].dict['status_nUp']
        status_frame = self.listTrajectories[0].dict['status_frame']
        sum_std = np.zeros(nT)
        for i in range(self.NB):
            sum_std += np.array(self.listTrajectories[i].dict['StdDev'])
        
        bestStd = np.zeros(nT, dtype = bool)
        i = 0
        while i < nT:
            if status_frame[i] == 0:
                bestStd[i] = True
                i += 1
            elif status_frame[i] > 0:
                s2 = status_nUp[i]
                L = [i]
                j = 0
                while i+j < nT-1 and status_nUp[i+j+1] == s2: # lazy evaluation of booleans
                    j += 1
                    L.append(i+j)
                #print(L)
                loc_std = sum_std[L]
                i_bestStd = i + int(np.argmax(loc_std))
                bestStd[i_bestStd] = True
                L = []
                i = i + j + 1

        return(bestStd)



    def computeForces(self, traj1, traj2, B0, D3, dx, k = K_CORRMAG):
        """
        """

        # Magnetization functions
        def computeMag_M270(B, k = k):
            M = k * 0.74257*1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
            return(M)

        def computeMag_M450(B, k = k):
            M = k * 1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
            return(M)

        dict_fMag = {'M270' : computeMag_M270, 'M450' : computeMag_M450}
        dictBeadTypes = {2.7 : 'M270', 4.5 : 'M450'}

        dictLogF = {'D3' : [], 'B0' : [], 'Btot_L' : [], 'Btot_R' : [], 
                    'F00' : [], 'F0' : [], 'dF_L' : [], 'dF_R' : [], 'Ftot' : []}

        # Correction functions
        def Bind_neighbour(B, D_BoI, neighbourType):
            if neighbourType == '' or neighbourType == 'nan':
                return(0)

            else:
                D_neighbour = self.beadDiameter
                fMag = dict_fMag[neighbourType] # Appropriate magnetization function
                M_neighbour = fMag(B) # magnetization [A.m^-1]
                V_neighbour = (4/3)*np.pi*(D_neighbour/2)**3 # volume [nm^3]
                m_neighbour = M_neighbour*V_neighbour*1e-9 # magnetic moment [A.nm^2]

                D_tot = (D_BoI + D_neighbour)/2 # Center-to-center distance [nm]
                B_ind = 2e5*m_neighbour/(D_tot**3) # Inducted mag field [mT]
                return(B_ind)

        def deltaF_neighbour(m_BoI, B, D_BoI, D_BoI2, neighbourType):
            if neighbourType == '' or neighbourType == 'nan':
                return(0)

            else:
                D_neighbour = self.beadDiameter
                fMag = dict_fMag[neighbourType] # Appropriate magnetization function
                M_neighbour = fMag(B) # magnetization [A.m^-1]
                V_neighbour = (4/3)*np.pi*(D_neighbour/2)**3 # volume [nm^3]
                m_neighbour = M_neighbour*V_neighbour*1e-9 # magnetic moment [A.nm^2]

                D_tot = D_BoI/2 + D_BoI2 + D_neighbour/2
                deltaF = 3e5*m_BoI*m_neighbour/D_tot**4 # force [pN]
                return(deltaF)

        # Let's make sure traj1 is the left bead traj and traj2 the right one.
        avgX1 = np.mean(traj1.dict['X'])
        avgX2 = np.mean(traj2.dict['X'])
        if avgX1 < avgX2:
            traj_L, traj_R = traj1, traj2
        else:
            traj_L, traj_R = traj2, traj1

        # Get useful data
        BeadType_L, BeadType_R = dictBeadTypes[traj_L.D], dictBeadTypes[traj_R.D]
        Neighbours_BL = np.concatenate(([traj_L.dict['Neighbour_L']], [traj_L.dict['Neighbour_R']]), axis = 0)
        Neighbours_BR = np.concatenate(([traj_R.dict['Neighbour_L']], [traj_R.dict['Neighbour_R']]), axis = 0)
        D_L, D_R = self.beadDiameter, self.beadDiameter

        nT = len(B0)
        D3nm = 1000*D3
        Dxnm = 1000*dx
        F = np.zeros(nT)

        # Maybe possible to process that faster on lists themselves
        for i in range(nT):
            # Appropriate magnetization functions
            f_Mag_L = dict_fMag[BeadType_L]
            f_Mag_R = dict_fMag[BeadType_R]

            # Btot = B0 + B inducted by potential left neighbour mag + B inducted by potential right neighbour mag
            Btot_L = B0[i] + Bind_neighbour(B0[i], D_L, Neighbours_BL[0,i]) + Bind_neighbour(B0[i], D_L, Neighbours_BL[1,i])
            Btot_R = B0[i] + Bind_neighbour(B0[i], D_R, Neighbours_BR[0,i]) + Bind_neighbour(B0[i], D_R, Neighbours_BR[1,i])

            # Magnetizations
            M_L = f_Mag_L(Btot_L)
            M_R = f_Mag_R(Btot_R)

            # Volumes
            V_L = (4/3)*np.pi*(D_L/2)**3 # volume [nm^3]
            V_R = (4/3)*np.pi*(D_R/2)**3 # volume [nm^3]

            # Magnetizations
            m_L = M_L * 1e-9 * V_L
            m_R = M_R * 1e-9 * V_R

            anglefactor = abs(3*(Dxnm[i]/D3nm[i])**2 - 1)

            # Forces
            F00 = 3e5*anglefactor * (f_Mag_L(B0[i])* 1e-9*V_L) * (f_Mag_R(B0[i])*1e-9*V_R) / (D3nm[i]**4)
            
            F0 = 3e5*anglefactor*m_L*m_R/D3nm[i]**4
            dF_L = deltaF_neighbour(m_L, B0[i], D_L, D_R, Neighbours_BR[1,i])
            dF_R = deltaF_neighbour(m_R, B0[i], D_R, D_L, Neighbours_BL[0,i])

            # Total force = force between beads involved in the pair (F0)
            #               + small force between B_L and B_R's potential right neighbour
            #               + small force between B_R and B_L's potential left neighbour
            F[i] = F0 + dF_L + dF_R

            dictLogF['D3'].append(D3nm[i]-(D_L+D_R)/2)
            dictLogF['B0'].append(B0[i])
            dictLogF['Btot_L'].append(Btot_L)
            dictLogF['Btot_R'].append(Btot_R)
            dictLogF['F00'].append(F00)
            dictLogF['F0'].append(F0)
            dictLogF['dF_L'].append(dF_L)
            dictLogF['dF_R'].append(dF_R)
            dictLogF['Ftot'].append(F[i])

        dfLogF = pd.DataFrame(dictLogF)

        return(F, dfLogF)


# %%%% Frame

class Frame:
    def __init__(self, F, iS, NB, Nup, status_frame, status_nUp, scale):
        ny, nx = F.shape[0], F.shape[1]
        self.F = F # Note : Frame.F points directly to the i-th frame of the image I ! To have 2 different versions one should use np.copy(F)
        self.NBoi = NB
        self.NBdetected = 0
        self.nx = nx
        self.ny = ny
        self.iS = iS
        self.listBeads = []
        self.trajPoint = []
        self.Nuplet = Nup
        self.status_frame = status_frame
        self.status_nUp = status_nUp
        self.scale = scale
        self.resDf = pd.DataFrame({'Area' : [], 'StdDev' : [], 'XM' : [], 'YM' : [], 'Slice' : []})

    def __str__(self):
        text = 'a'
        return(text)

    def show(self, strech = True):
        fig, ax = plt.subplots(1,1)
#         fig_size = plt.gcf().get_size_inches()
#         fig.set_size_inches(2 * fig_size)
        if strech:
            pStart, pStop = np.percentile(self.F, (1, 99))
            ax.imshow(self.F, cmap = 'gray', vmin = 0, vmax = 3000)
        else:
            ax.imshow(self.F, cmap = 'gray')
        if len(self.listBeads) > 0:
            for B in self.listBeads:
                ax.plot([B.x], [B.y], c='orange', marker='+', markersize = 15)
        fig.show()

    def makeListBeads(self):
        self.NBdetected = self.resDf.shape[0]
        for i in range(self.NBdetected):
            d = {}
            for c in self.resDf.columns:
                d[c] = self.resDf[c].values[i]
            self.listBeads.append(Bead(d, self.F))

    def beadsXYarray(self):
        A = np.zeros((len(self.listBeads), 2))
        for i in range(len(self.listBeads)):
            b = self.listBeads[i]
            A[i,0], A[i,1] = b.x, b.y
        return(A)

    def beadsStdDevarray(self):
        A = np.zeros(len(self.listBeads))
        for i in range(len(self.listBeads)):
            b = self.listBeads[i]
            A[i] = b.std
        return(A)




# %%%% Bead

class Bead:
    def __init__(self, d, F):
        self.x = d['XM']
        self.y = d['YM']
        self.D = 0
        self.area = d['Area']
        self.std = d['StdDev']
        self.iS = d['Slice']-1
        self.status_frame = ''
        self.Neighbour_L = ''
        self.Neighbour_R = ''
        self.F = F


    def show(self, strech = True):
        fig, ax = plt.subplots(1,1)
        if strech:
            pStart, pStop = np.percentile(self.F, (1, 99))
            ax.imshow(self.F, cmap = 'gray', vmin = pStart, vmax = pStop)
        else:
            ax.imshow(self.F, cmap = 'gray')
        ax.plot([self.x], [self.y], c='orange', marker='o')
        fig.show()

#

# %%%% Trajectory

class Trajectory:
    def __init__(self, I, cellID, listFrames, scale, Zstep, iB):
        nS, ny, nx = I.shape[0], I.shape[1], I.shape[2]
        self.I = I
        self.cellID = cellID
        self.listFrames = listFrames
        self.scale = scale
        self.nx = nx
        self.ny = ny
        self.nS = nS
        self.D = 0
        self.nT = 0
        self.iB = iB
        self.dict = {'X': [],'Y': [],'idxAnalysis': [],'StdDev': [],
                     'Bead': [],'status_frame': [],'status_nUp': [],'iF': [],'iS': [],'iB_inFrame' : [],
                     'bestStd' : [], 'Zr' : [], 'Neighbour_L' : [], 'Neighbour_R' : []}
        # iF is the index in the listFrames
        # iS is the index of the slice in the raw image MINUS ONE
        self.beadInOut = ''
        self.deptho = []
        self.depthoPath = ''
        self.depthoStep = 20
        self.depthoZFocus = 200
        self.Zstep = Zstep # The step in microns between 2 consecutive frames in a multi-frame Nuplet
        
        #### Z detection settings here
        self.HDZfactor = 5
        self.maxDz_triplets = 60 # Max Dz allowed between images
        self.maxDz_singlets = 30
        self.HWScan_triplets = 1200 # Half width of the scans
        self.HWScan_singlets = 600
        
    def __str__(self):
        text = 'iS : ' + str(self.series_iS)
        text += '\n'
        text += 'XY : ' + str(self.seriesXY)
        return(text)

    def save(self, path):
        df = pd.DataFrame(self.dict)
        df.to_csv(path, sep = '\t', index = False)

    def computeZ(self, matchingDirection, plot = 0):
        

        if len(self.deptho) == 0:
            return('Error, no depthograph associated with this trajectory')

        else:
            Ddz, Ddx = self.deptho.shape[0], self.deptho.shape[1]
            iF = self.dict['iF'][0]
            previousZ = -1
            
            
            while iF <= max(self.dict['iF']):
                
            #### Enable plots of Z detection  here
                # plot = 0
                # if (iF >= 0 and iF <= 20):# or (iF > 400 and iF <= 440):
                #     plot = 1
                    
            # ###################################################################

                if iF not in self.dict['iF']: # this index isn't in the trajectory list => the frame was removed for some reason.
                    iF += 1 # Let's just go to the next index

                else:
                    F = self.listFrames[iF]
                    Nup = F.Nuplet
                    if Nup <= 1:
                        framesNuplet = [F]
                        iFNuplet = [iF]
                        iF += 1
                    elif Nup > 1:
                        framesNuplet = [F]
                        iFNuplet = [iF]
                        jF = 1
                        while iF+jF <= max(self.dict['iF']) and self.listFrames[iF+jF].status_nUp == F.status_nUp:
                            if iF+jF in self.dict['iF']: # One of the images of the triplet may be invalid,
                                # and we don't want to take it. With this test we won't
                                nextF = self.listFrames[iF+jF]
                                framesNuplet.append(nextF)
                                iFNuplet.append(iF+jF)
                            jF += 1

                        iF += jF

                    Z = self.findZ_Nuplet(framesNuplet, iFNuplet, Nup, previousZ, 
                                          matchingDirection, plot)
                    previousZ = Z
                    # This Z_pix has no meaning in itself, it needs to be compared to the depthograph Z reference point,
                    # which is depthoZFocus.

                    Zr = self.depthoZFocus - Z # If you want to find it back, Z = depthoZFocus - Zr
                    # This definition was chosen so that when Zr > 0, the plane of observation of the bead is HIGHER than the focus
                    # and accordingly when Zr < 0, the plane of observation of the bead is LOWER than the focus

                    mask = np.array([(iF in iFNuplet) for iF in self.dict['iF']])
                    self.dict['Zr'][mask] = Zr



    def findZ_Nuplet(self, framesNuplet, iFNuplet, Nup, previousZ, 
                     matchingDirection, plot):
        Nframes = len(framesNuplet)
        listStatus_1 = [F.status_frame for F in framesNuplet]
        listXY = [[self.dict['X'][np.where(self.dict['iF']==iF)][0],
                   self.dict['Y'][np.where(self.dict['iF']==iF)][0]] for iF in iFNuplet]
        listiS = [self.dict['iS'][np.where(self.dict['iF']==iF)][0] for iF in iFNuplet]
        cleanSize = ufun.getDepthoCleanSize(self.D, self.scale)
        hdSize = self.deptho.shape[1]
        depthoDepth = self.deptho.shape[0]
        listProfiles = np.zeros((Nframes, hdSize))
        listROI = []
        listWholeROI = []
        for i in range(Nframes):
            xx = np.arange(0, 5)
            yy = np.arange(0, cleanSize)
            try:
                
                X, Y = listXY[i][0], listXY[i][1]
                roughSize = cleanSize + 4
                roughCenter = roughSize // 2
                xb1, yb1, xb2, yb2, validROI = ufun.getROI(roughSize, X, Y, self.nx, self.ny)
                F_roughRoi = framesNuplet[i].F[yb1:yb2, xb1:xb2]
                
                xc1, yc1 = X-xb1-0.5, Y-yb1-0.5
                translation = (xc1-roughCenter, yc1-roughCenter)
                
                tform = transform.EuclideanTransform(rotation=0, translation = translation)
                F_tmp = transform.warp(F_roughRoi, tform, order = 1, preserve_range = True)

                wholeROI = np.copy(F_tmp[roughCenter-cleanSize//2:roughCenter+cleanSize//2+1,\
                                           roughCenter-cleanSize//2:roughCenter+cleanSize//2+1])
                cleanCenter = cleanSize // 2
                
                
                profileROI = wholeROI[:, cleanCenter-2:cleanCenter+3] # line that is 5 pixels wide     
                profileROI_hd = ufun.resize_2Dinterp(profileROI, new_nx = 5, new_ny = hdSize)
                
                # X, Y, iS = int(np.round(listXY[i][0])), int(np.round(listXY[i][1])), listiS[i] # > We could also try to recenter the image to keep a subpixel resolution here
                # # line that is 5 pixels wide
                # wholeROI = framesNuplet[i].F[Y-cleanSize//2:Y+cleanSize//2+1, X-cleanSize//2:X+cleanSize//2+1]
                # profileROI = framesNuplet[i].F[Y-cleanSize//2:Y+cleanSize//2+1, X-2:X+3]
                # f = interpolate.interp2d(xx, yy, profileROI, kind='cubic')
                # # Now use the obtained interpolation function and plot the result:
                # xxnew = xx
                # yynew = np.linspace(0, cleanSize, hdSize)
                # profileROI_hd = f(xxnew, yynew)

            except: # If the vertical slice doesn't work, try the horizontal one
                print(gs.ORANGE + 'error with the vertical slice -> trying with horizontal one')
                print('' + gs.NORMAL)

                xx, yy = yy, xx
                X, Y, iS = int(np.round(listXY[i][0])), int(np.round(listXY[i][1])), listiS[i] # > We could also try to recenter the image to keep a subpixel resolution here
                # line that is 5 pixels wide
                wholeROI = framesNuplet[i].F[Y-cleanSize//2:Y+cleanSize//2+1, X-cleanSize//2:X+cleanSize//2+1]
                profileROI = framesNuplet[i].F[Y-2:Y+3, X-cleanSize//2:X+cleanSize//2+1]
                # Now use the obtained interpolation function and plot the result:
                profileROI_hd = ufun.resize_2Dinterp(profileROI, new_nx = hdSize, new_ny = 5).T

            listROI.append(profileROI)
            listWholeROI.append(wholeROI)

            listProfiles[i,:] = profileROI_hd[:,5//2] * (1/5)
            for j in range(1, 1 + 5//2):
                listProfiles[i,:] += profileROI_hd[:,5//2-j] * (1/5)
                listProfiles[i,:] += profileROI_hd[:,5//2+j] * (1/5)

        listProfiles = listProfiles.astype(np.uint16)



        # now use listStatus_1, listProfiles, self.deptho + data about the jump between Nuplets ! (TBA)
        # to compute the correlation function
        nVoxels = int(np.round(int(self.Zstep)/self.depthoStep))
        
        if previousZ == -1:
            Ztop = 0
            Zbot = depthoDepth
        
        elif Nup > 1:
            HW = self.HWScan_triplets
            halfScannedDepth_raw = int(HW / self.depthoStep)
            Ztop = max(0, previousZ - halfScannedDepth_raw) 
            Zbot = min(depthoDepth, previousZ + halfScannedDepth_raw)
            
        elif Nup == 1:
            HW = self.HWScan_singlets
            halfScannedDepth_raw = int(HW / self.depthoStep) 
            Ztop = max(0, previousZ - halfScannedDepth_raw) 
            Zbot = min(depthoDepth, previousZ + halfScannedDepth_raw)

        scannedDepth = Zbot-Ztop
        # print(Nup, depthoDepth, Ztop, Zbot, scannedDepth)
        
        listDistances = np.zeros((Nframes, scannedDepth))
        listZ = np.zeros(Nframes, dtype = int)
        Zscanned = np.arange(Ztop,Zbot,1, dtype=int)
        
        subDeptho = self.deptho[Ztop:Zbot,:]
        
        for i in range(Nframes):
            
            listDistances[i] = ufun.squareDistance(subDeptho, listProfiles[i], normalize = True) # Utility functions
            listZ[i] = Ztop + np.argmin(listDistances[i])

        # Translate the profiles that must be translated (status_frame 1 & 3 if Nup = 3)
        # and don't move the others (status_frame 2 if Nup = 3 or the 1 profile when Nup = 1)
        if Nup > 1:
            finalDists = ufun.matchDists(listDistances, listStatus_1, Nup, 
                                        nVoxels, direction = matchingDirection)
        elif Nup == 1:
            finalDists = listDistances

        sumFinalD = np.sum(finalDists, axis = 0)


        #### Tweak this part to force the Z-detection to a specific range to prevent abnormal jumps
        if previousZ == -1: # First image => No restriction
            Z = np.argmin(sumFinalD)
            maxDz = 0
            
        else: # Not first image => Restriction
            if Nup > 1 and previousZ != -1: # Not first image AND Triplets => Restriction Triplets
                maxDz = self.maxDz_triplets
            elif Nup == 1 and previousZ != -1: # Not first image AND singlet => Restriction Singlet
                maxDz = self.maxDz_singlets
                
            limInf = max(previousZ - maxDz, 0) - Ztop
            limSup = min(previousZ + maxDz, depthoDepth) - Ztop
            Z = Ztop + limInf + np.argmin(sumFinalD[limInf:limSup])
            
            
            
        #### Important plotting option here
        if plot >= 1:

            plt.ioff()
            fig, axes = plt.subplots(5, 3, figsize = (16,16))
            
            cmap = 'magma'
            color_image = 'cyan'
            color_Nup = ['gold', 'darkorange', 'red']
            color_result = 'darkgreen'
            color_previousResult = 'turquoise'
            color_margin = 'aquamarine'
            
            im = framesNuplet[0].F
            X2, Y2 = listXY[0][0], listXY[0][1]
            
            deptho_zticks_list = np.arange(0, depthoDepth, 50*self.HDZfactor, dtype = int)
            deptho_zticks_loc = ticker.FixedLocator(deptho_zticks_list)
            deptho_zticks_format = ticker.FixedFormatter((deptho_zticks_list/self.HDZfactor).astype(int))

            
            if Nup == 1:
                direction = 'Single Image'
            else:
                direction = matchingDirection

            pStart, pStop = np.percentile(im, (1, 99))
            axes[0,0].imshow(im, vmin = pStart, vmax = 1.5*pStop, cmap = 'gray')
            images_ticks_loc = ticker.MultipleLocator(50)
            axes[0,0].xaxis.set_major_locator(images_ticks_loc)
            axes[0,0].yaxis.set_major_locator(images_ticks_loc)
            
            
            dx, dy = 50, 50
            axes[0,0].plot([X2], [Y2], marker = '+', c = 'red')
            axes[0,0].plot([X2-dx,X2-dx], [Y2-dy,Y2+dy], ls = '--', c = color_image, lw = 0.8)
            axes[0,0].plot([X2+dx,X2+dx], [Y2-dy,Y2+dy], ls = '--', c = color_image, lw = 0.8)
            axes[0,0].plot([X2-dx,X2+dx], [Y2-dy,Y2-dy], ls = '--', c = color_image, lw = 0.8)
            axes[0,0].plot([X2-dx,X2+dx], [Y2+dy,Y2+dy], ls = '--', c = color_image, lw = 0.8)
            # axes[0,0].set_xlim([X2-dx-1,X2+dx+2])
            # axes[0,0].set_ylim([Y2+dy+1, Y2-dy-2])

            # Plot the deptho then resize it better
            axes[0,1].imshow(self.deptho, cmap = cmap)
            XL0, YL0 = axes[0,1].get_xlim(), axes[0,1].get_ylim()
            extent = (XL0[0], YL0[0]*(5/3), YL0[0], YL0[1])
            axes[0,1].imshow(self.deptho, extent = extent, cmap = cmap)
            
            axes[0,1].yaxis.set_major_locator(deptho_zticks_loc)
            axes[0,1].yaxis.set_major_formatter(deptho_zticks_format)
            
            pixLineHD = np.arange(0, hdSize, 1)
            zPos = Zscanned
            
            
            for i in range(Nframes):
                status_frame = int(framesNuplet[i].status_frame)
                status_frame += (status_frame == 0)
                
                # Show the bead appearence
                axes[1,i].imshow(listWholeROI[i], cmap = cmap)
                images_ticks_loc = ticker.MultipleLocator(10)
                axes[1,i].xaxis.set_major_locator(images_ticks_loc)
                axes[1,i].yaxis.set_major_locator(images_ticks_loc)
                axes[1,i].set_title('Image {:.0f}/{:.0f} - '.format(status_frame, Nup) + direction, 
                                    fontsize = 14)
                axes[1,i].axvline(cleanSize//2, c=color_Nup[i], ls='--', lw = 1)
                
                # #### HERE PLOT OPTION
                axes[1,i].set_xlim([cleanSize//2-10,cleanSize//2+10])
                axes[1,i].set_ylim([cleanSize//2+10,cleanSize//2-10])
                
                # Show the profile of the beads
                axes[2,i].plot(pixLineHD, listProfiles[i], c = color_Nup[i])
                axes[2,i].set_xlabel('Position along the profile - (Y-axis)', 
                                      fontsize = 9)
                axes[2,i].set_ylabel('Pixel intensity', 
                                      fontsize = 9)
                axes[2,i].set_title('Profile {:.0f}/{:.0f} - '.format(status_frame, Nup), 
                                    fontsize = 11)
                
                # Show the distance map to the deptho
                listDistances = np.array(listDistances)
                # inversed_listDistances = (listDistances[i] * (-1)) + np.max(listDistances[i])
                # peaks, peaks_prop = signal.find_peaks(inversed_listDistances, distance = self.HDZfactor * 20)
                axes[3,i].plot(zPos, listDistances[i])
                # axes[3,i].plot(zPos, inversed_listDistances, ls='--', lw=0.75, c='k')
                axes[3,i].xaxis.set_major_locator(deptho_zticks_loc)
                axes[3,i].xaxis.set_major_formatter(deptho_zticks_format)
                limy3 = axes[3,i].get_ylim()
                min_i = zPos[np.argmin(listDistances[i])]
                axes[3,i].plot([min_i, min_i], limy3, ls = '--', c = color_Nup[i])
                
                axes[3,i].set_xlabel('Position along the depthograph - (Z-axis)', 
                                      fontsize = 9)
                axes[3,i].set_ylabel('Cost\n(Squared diff to deptho)', 
                                      fontsize = 9)
                pos_nm = min_i/self.HDZfactor
                axes[3,i].set_title(f'Cost curve {status_frame:.0f}/{Nup:.0f} - pos = {pos_nm:.0f}', 
                                    fontsize = 11)
                
                # for p in peaks:
                #     p_i = zPos[int(p)]
                #     axes[3,i].plot([p_i], [np.mean(limy3)], ls = '',
                #                   marker = 'v',  c = 'orange', mec = 'k', markersize = 8)
                #     axes[3,i].text(p_i, np.mean(limy3)*1.1, str(p_i/self.HDZfactor), c = 'k')
                axes[3,i].set_xlim([0, depthoDepth])
                
                
                
                #
                axes[4,i].plot(zPos, finalDists[i])
                axes[4,i].xaxis.set_major_locator(deptho_zticks_loc)
                axes[4,i].xaxis.set_major_formatter(deptho_zticks_format)
                limy4 = axes[4,i].get_ylim()
                min_i = zPos[np.argmin(finalDists[i])]
                axes[4,i].plot([min_i, min_i], limy4, ls = '--', c = color_Nup[i])
                
                axes[4,i].set_xlabel('Corrected position along the depthograph - (Z-axis)', 
                                      fontsize = 9)
                axes[4,i].set_ylabel('Cost\n(Squared diff to deptho)', 
                                      fontsize = 9)
                pos_nm = min_i/self.HDZfactor
                axes[4,i].set_title(f'Cost curve with corrected position {status_frame:.0f}/{Nup:.0f} - pos = {pos_nm:.0f}',
                                    fontsize = 11)
                
                # axes[4,i].text(min_i+5, np.mean(limy4), str(min_i/self.HDZfactor), c = 'k')
                axes[4,i].set_xlim([0, depthoDepth])



                axes[0,1].plot([axes[0,1].get_xlim()[0], axes[0,1].get_xlim()[1]-1], 
                                [listZ[i], listZ[i]], 
                                ls = '--', c = color_Nup[i])
                
                axes[0,1].plot([axes[0,1].get_xlim()[0], axes[0,1].get_xlim()[1]-1], 
                                [Z,Z], 
                                ls = '--', c = color_result)


            axes[0,2].plot(zPos, sumFinalD)
            axes[0,2].xaxis.set_major_locator(deptho_zticks_loc)
            axes[0,2].xaxis.set_major_formatter(deptho_zticks_format)
            limy0 = axes[0,2].get_ylim()
            axes[0,2].plot([Z, Z], limy0, ls = '-', c = color_result, label = 'Z', lw = 1.5)
            axes[0,2].plot([previousZ, previousZ], limy0, 
                            ls = '--', c = color_previousResult, label = 'previous Z', lw = 0.8)
            axes[0,2].plot([previousZ-maxDz, previousZ-maxDz], limy0,
                            ls = '--', c = color_margin, label = 'allowed margin', lw = 0.8)
            axes[0,2].plot([previousZ+maxDz, previousZ+maxDz], limy0,
                            ls = '--', c = color_margin, lw = 0.8)
            axes[0,2].set_xlim([0, depthoDepth])
            
            axes[0,2].set_xlabel('Position along the depthograph\n(Z-axis)', 
                                  fontsize = 9)
            axes[0,2].set_ylabel('Total Cost\n(Sum of Squared diff to deptho)', 
                                  fontsize = 9)
            axes[0,2].set_title('Sum of Cost curves with corrected position', 
                                fontsize = 11)
            axes[0,2].legend()
            
            for ax in axes.flatten():
                ax.tick_params(axis='x', labelsize=9)
                ax.tick_params(axis='y', labelsize=9)
            
            Nfig = plt.gcf().number
            iSNuplet = [F.iS for F in framesNuplet]
            
            fig.tight_layout()
            fig.subplots_adjust(top=0.94)
            
            fig.suptitle('Frames '+str(iFNuplet)+' - Slices '+str(iSNuplet)+' ; '+\
                          'Z = {:.1f} slices = '.format(Z/self.HDZfactor) + \
                          '{:.4f} µm'.format(Z*(self.depthoStep/1000)),
                          y=0.98)
                
            ZPlots_path = f'.\\Zplots\\{self.cellID}\\'
            if not os.path.isdir('.\\Zplots\\'):
                os.mkdir('.\\Zplots\\')
            if not os.path.isdir(ZPlots_path):
                os.mkdir(ZPlots_path)
            
            saveName = 'ZCheckPlot_S{:.0f}_B{:.0f}.png'.format(iSNuplet[0], self.iB+1)
            savePath = os.path.join(ZPlots_path, saveName)
            fig.savefig(savePath)
            plt.close(fig)
        
        plt.ion()
            
            
            
        return(Z)



    def keepBestStdOnly(self):
        dictBestStd = {}
        bestStd = self.dict['bestStd']
        nT = int(np.sum(bestStd))
        for k in self.dict.keys():
            A = np.array(self.dict[k])
            dictBestStd[k] = A[bestStd]
        self.dict = dictBestStd
        self.nT = nT

    def askUi(self, beadType):
        # Plots to help the user to see the neighbour of each bead
        Nimg = 4
        frequency = self.nS/(Nimg-1)
        fig, ax = plt.subplots(2, 2)
        axes = ax.flatten()
        for i in range(Nimg):
            # try:
            ii = int(min(i*frequency, self.nS-2))
            pos = np.searchsorted(self.dict['iS'], ii, 'left')
            iS = self.dict['iS'][pos]
            iF = self.dict['iF'][pos]
            pStart, pStop = np.percentile(self.I[iS], (1, 99))
            axes[i].imshow(self.I[iS], cmap = 'gray', vmin = pStart, vmax = pStop)
            axes[i].set_title('Image ' + str(iS) + ' / ' + str(self.nS), fontsize = 12)
            axes[i].plot([self.dict['X'][pos]],[self.dict['Y'][pos]], 'ro')
            # except:
            #     print(gs.RED  + 'error in askUi' + gs.NORMAL)
                
        
        # Display the figure    
        plt.show()
        
        #### Windows specific
#         mngr = plt.get_current_fig_manager()
#         mngr.window.setGeometry(720, 50, 1175, 1000)
        
        # Ask the question(s)
        # Q1
        bead_pos = pyautogui.confirm(
            text='Is it an inside or outside bead?',
            title='',
            buttons=['In', 'Out'])
        
        # Q2
        bead_neigh = pyautogui.confirm(
            text='Neighbours of the selected bead?',
            title='',
            buttons=['1', '2'])
        
        # Stop displaying the figure 
        plt.close(fig)
        
        # Save the question's answer:
            # A1
        self.beadInOut = bead_pos
         
            # A2
        if bead_neigh == '1':
            if self.iB%2 == 0: # the bead is on the left of a pair
                Neighbour_L, Neighbour_R = '', beadType
            elif self.iB%2 == 1: # the bead is on the right of a pair
                Neighbour_L, Neighbour_R = beadType, ''
        elif bead_neigh == '2':
            Neighbour_L, Neighbour_R = beadType, beadType

        listNeighbours = []
        for i in range(len(self.dict['iF'])):
            self.dict['Bead'][i].Neighbour_L = Neighbour_L
            self.dict['Bead'][i].Neighbour_R = Neighbour_R
            listNeighbours.append([Neighbour_L, Neighbour_R])

        arrayNeighbours = np.array(listNeighbours)
        self.dict['Neighbour_L'] = arrayNeighbours[:,0]
        self.dict['Neighbour_R'] = arrayNeighbours[:,1]


    def plot(self, ax, i_color):
        colors = gs.colorList10
        c = colors[i_color]
        ax.plot(self.dict['X'], self.dict['Y'], color=c, lw=1)

# %%%% Main

# def mainTracker(dates, manips, wells, cells, depthoNames, expDf, NB = 2,
#                 sourceField = 'default', redoAllSteps = False, trackAll = False,
#                 DirData = cp.DirData, 
#                 DirDataRaw = cp.DirDataRaw, 
#                 DirDataRawDeptho = cp.DirDataRawDeptho, 
#                 DirDataTimeseries = cp.DirDataTimeseries,
#                 CloudSaving = cp.CloudSaving,
#                 DirCloudTimeseries = cp.DirCloudTimeseries):
    
def mainTracker(dictPaths, dictConstants, **kwargs):
    
    start = time.time()

    #### 0. Load different data sources & Preprocess
    
    PathRawDataDir = dictPaths['PathRawDataDir']
    PathDeptho = dictPaths['PathDeptho']
    PathResultsDir = dictPaths['PathResultsDir']
    
    dictOptions = {'redoAllSteps' : False, 
                     'timepoints' : True
                     }
    
    dictOptions.update(kwargs)
    
    timePoints = dictOptions['timepoints']
    redoAllSteps = dictOptions['redoAllSteps']
    trackAll = False
    
    #### 0.1 Create some new folders if necessary
    trajDirRaw = os.path.join(PathResultsDir, 'Trajectories_raw')
    if not os.path.exists(trajDirRaw):
        os.mkdir(trajDirRaw)
        
    trajDir = os.path.join(PathResultsDir, 'Trajectories')
    if not os.path.exists(trajDir):
        os.mkdir(trajDir)
    
    #### 0.2 - Make list of files to analyse

    imagesToAnalyse = []
    imagesToAnalyse_Paths = []

    fileList = os.listdir(PathRawDataDir)
    for f in fileList:
        fPath = os.path.join(PathRawDataDir, f)
        if os.path.isfile(fPath[:-4] + '_Results.txt'):
            if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.TIF'):
                imagesToAnalyse.append(f)
                imagesToAnalyse_Paths.append(fPath)    
    
    
    #### 0.3 - Begining of the Main Loop
    for i in range(len(imagesToAnalyse)): 
        f, fP = imagesToAnalyse[i], imagesToAnalyse_Paths[i]
        cellID = '.'.join(f.split('.')[:-1])

        print('\n')
        print(gs.BLUE + 'Analysis of file {:.0f}/{:.0f} : {}'.format(i+1, len(imagesToAnalyse), f))
        print('Loading image and experimental data...' + gs.NORMAL)
    

        #### 0.4 - Load image and init PTL
        I = io.imread(fP) # Approx 0.5s per image
        PTL = PincherTimeLapse(I, cellID, dictConstants)
    
        #### 0.5 - Load field file
        timeFilePath = fP[:-4] + '_Timepoints.txt'
        if timePoints == True:
            fieldCols = ['T_abs']
            fieldDf = pd.read_csv(timeFilePath, sep = '\t', names = fieldCols) # '\t'
            N = len(fieldDf.T_abs.values)
            fieldDf['B_set'] = PTL.MagField * np.ones(N, dtype = np.float64)
        else:    
            N = PTL.nS
            T_abs = np.arange(N, dtype = np.float64)
            B_set = PTL.MagField * np.ones(N, dtype = np.float64)
            dict_field = {'T_abs':T_abs,
                          'B_set':B_set}
            fieldDf = pd.DataFrame(dict_field)
            
        #### 0.6 - Check if a log file exists and load it if required
        logFilePath = fP[:-4] + '_LogPY.txt'
        logFileImported = False
        if redoAllSteps:
            pass
        else:    
            if os.path.isfile(logFilePath):
                PTL.importLog(logFilePath)
                PTL.dictLog['UILog'] = PTL.dictLog['UILog'].astype(str)
                logFileImported = True

        print(gs.BLUE + 'OK!')
        print(gs.BLUE + 'Pretreating the image...' + gs.NORMAL)

        #### 0.8 - Sort slices
        if not logFileImported:
            PTL.determineFramesStatus()

        PTL.saveLog(display = False, save = (not logFileImported), path = logFilePath)

        #### 0.11 - Create list of Frame objects
        PTL.makeFramesList()
        

        print(gs.BLUE + 'OK!' + gs.NORMAL)


    #### 1. Detect beads

        print(gs.BLUE + 'Detecting all the bead objects...' + gs.NORMAL)
        Td = time.time()

        #### 1.1 - Import the _Results.txt file
        resFilePath = fP[:-4] + '_Results.txt'
        PTL.importBeadsDetectResult(resFilePath)
        

        #### 1.2 - Detect the beads
        # Input the results in each Frame objects [the results have been loaded at the previous step]
        PTL.detectBeads()
            
        print(gs.BLUE + 'OK! dT = {:.3f}'.format(time.time()-Td) + gs.NORMAL)


    #### 2. Make trajectories for beads of interest
        # One of the main steps ! The tracking of the beads happens here !

        print(gs.BLUE + 'Tracking the beads of interest...' + gs.NORMAL)
        Tt = time.time()

        #### 2.1 - Check if some trajectories exist already
        
            
        # trajFilesExist_global = False
        trajFilesImported = False
        trajFilesExist_sum = 0
        
        if redoAllSteps:
            pass
        
        else:
            allTrajPaths = [os.path.join(trajDirRaw, f[:-4] + '_rawTraj' + str(iB) + '' + '_PY.csv') for iB in range(PTL.NB)]
            allTrajPaths += [os.path.join(trajDirRaw, f[:-4] + '_rawTraj' + str(iB) + '_In' + '_PY.csv') for iB in range(PTL.NB)]
            allTrajPaths += [os.path.join(trajDirRaw, f[:-4] + '_rawTraj' + str(iB) + '_Out' + '_PY.csv') for iB in range(PTL.NB)]
            allTrajPaths = np.array(allTrajPaths)
            trajFilesExist = np.array([os.path.isfile(trajPath) for trajPath in allTrajPaths])
            trajFilesExist_sum = np.sum(trajFilesExist)

        #### 2.2 - If yes, load them
        if trajFilesExist_sum == PTL.NB:
            trajFilesImported = True
            trajPaths = allTrajPaths[trajFilesExist]
            for iB in range(PTL.NB):
                PTL.importTrajectories(trajPaths[iB], iB)
            print(gs.GREEN + 'Raw traj files found and imported :)' + gs.NORMAL)

        #### 2.3 - If no, compute them by tracking the beads
        if not trajFilesImported:
            outcome = PTL.buildTrajectories(trackAll = trackAll) 
            
            # Main tracking function !
            if outcome == 'Bug':
                continue
            else:
                pass
        
        #### Optional useful plot
        # PTL.plot_computed_trajectories()

        #### 2.4 - Save the user inputs
        PTL.saveLog(display = 0, save = True, path = logFilePath)

        print(gs.BLUE + 'OK! dT = {:.3f}'.format(time.time()-Tt) + gs.NORMAL)

        #### 2.5 - Sort the trajectories [Maybe unnecessary]


    #### 3. Qualify - Detect boi sizes and neighbours

        #### 3.1 - Set BOI size

        if 'M450' in PTL.beadType:
            D = 4.5
        elif 'M270' in PTL.beadType:
            D = 2.7

        first_iF = PTL.listTrajectories[0].dict['iF'][0]
        for B in PTL.listFrames[first_iF].listBeads:
            B.D = D

        # Propagate it across the trajectories
        for iB in range(PTL.NB):
            traj = PTL.listTrajectories[iB]
            B0 = traj.dict['Bead'][0]
            D = B0.D
            traj.D = D
            for B in traj.dict['Bead']:
                B.D = D

        #### 3.2 - Call for user input to detect some stuff

        # Current way, with user input
        if redoAllSteps or not trajFilesImported:
            for iB in range(PTL.NB):
                traj = PTL.listTrajectories[iB]
                beadType = PTL.beadType
                traj.askUi(beadType = beadType)



    #### 4. Compute dz

        #### 4.1 - Import depthographs
        HDZfactor = PTL.listTrajectories[0].HDZfactor
        
        depthoPath = PathDeptho
        deptho_root = '.'.join(depthoPath.split('.')[:-1])
        deptho = io.imread(depthoPath)
        depthoMetadata = pd.read_csv(deptho_root + '_Metadata.csv', sep=';')
        depthoStep = depthoMetadata.loc[0, 'step']
        depthoZFocus = depthoMetadata.loc[0, 'focus']

        # increase the resolution of the deptho with interpolation
        # print('deptho shape check')
        # print(deptho.shape)
        nX, nZ = deptho.shape[1], deptho.shape[0]
        XX, ZZ = np.arange(0, nX, 1), np.arange(0, nZ, 1)
        # print(XX.shape, ZZ.shape)
        fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
        ZZ_HD = np.arange(0, nZ, 1/HDZfactor)
        # print(ZZ_HD.shape)
        depthoHD = fd(XX, ZZ_HD)
        depthoStepHD = depthoStep/HDZfactor
        depthoZFocus = depthoZFocus*HDZfactor
        # print(depthoHD.shape)
        #
        for iB in range(PTL.NB):
            traj = PTL.listTrajectories[iB]
            traj.deptho = depthoHD
            traj.depthoPath = depthoPath
            traj.depthoStep = depthoStepHD
            traj.depthoZFocus = depthoZFocus
            traj.HDZfactor = HDZfactor


        #### 4.2 - Compute z for each traj
        matchingDirection = PTL.Zdirection
        print(gs.ORANGE + "Deptho detection '" + str(matchingDirection) + "' mode" + gs.NORMAL)
        
        if redoAllSteps or not trajFilesImported:
            for iB in range(PTL.NB):
                # np.set_printoptions(threshold=np.inf)
                print(gs.BLUE + 'Computing Z in traj  {:.0f}...'.format(iB+1) + gs.NORMAL)
                Tz = time.time()
                traj = PTL.listTrajectories[iB]
                traj.computeZ(matchingDirection, plot = 0)
                print(gs.BLUE + 'OK! dT = {:.3f}'.format(time.time()-Tz) + gs.NORMAL)

        else:
            print(gs.BLUE + 'Computing Z...' + gs.NORMAL)
            print(gs.GREEN + 'Z had been already computed :)' + gs.NORMAL)
        

        #### 4.3 - Save the raw traj (before Std selection)
        if redoAllSteps or not trajFilesImported:
            for iB in range(PTL.NB):
                traj = PTL.listTrajectories[iB]
                traj_df = pd.DataFrame(traj.dict)
                trajPathRaw = os.path.join(trajDirRaw, f[:-4] + '_rawTraj' + str(iB) + '_' + traj.beadInOut + '_PY.csv')
                traj_df.to_csv(trajPathRaw, sep = '\t', index = False)

        #### 4.4 - Keep only the best std data in the trajectories
        for iB in range(PTL.NB):
            traj = PTL.listTrajectories[iB]
            traj.keepBestStdOnly()

        #### 4.5 - The trajectories won't change from now on. We can save their '.dict' field.
        if redoAllSteps or not trajFilesImported:
            for iB in range(PTL.NB):
                traj = PTL.listTrajectories[iB]
                traj_df = pd.DataFrame(traj.dict)
                trajPath = os.path.join(trajDir, f[:-4] + '_traj' + str(iB) + '_' + traj.beadInOut + '_PY.csv')
                traj_df.to_csv(trajPath, sep = '\t', index = False)
                
    
    
    #### 5. Define pairs and compute distances
        print(gs.BLUE + 'Computing distances...' + gs.NORMAL)

        #### 5.1 - In case of 1 pair of beads
        if PTL.NB == 2:
            traj1 = PTL.listTrajectories[0]
            traj2 = PTL.listTrajectories[1]
            nT = traj1.nT

            #### 5.1.1 - Create a dict to prepare the export of the results
            timeSeries = {
                'idxAnalysis' : np.zeros(nT),
                'T' : np.zeros(nT),
                'Tabs' : np.zeros(nT),
                'B' : np.zeros(nT),
                'F' : np.zeros(nT),
                'dx' : np.zeros(nT),
                'dy' : np.zeros(nT),
                'dz' : np.zeros(nT),
                'D2' : np.zeros(nT),
                'D3' : np.zeros(nT),
            }

            #### 5.1.2 - Input common values:
            T0 = fieldDf['T_abs'].values[0]/1000 # From ms to s conversion
            timeSeries['idxAnalysis'] = traj1.dict['idxAnalysis']
            timeSeries['Tabs'] = (fieldDf['T_abs'][traj1.dict['iField']])/1000 # From ms to s conversion
            timeSeries['T'] = timeSeries['Tabs'].values - T0*np.ones(nT)
            timeSeries['B'] = fieldDf['B_set'][traj1.dict['iField']].values
            timeSeries['B'] *= PTL.MagCorrFactor

            #### 5.1.3 - Compute distances
            timeSeries['dx'] = (traj2.dict['X'] - traj1.dict['X'])/PTL.scale
            timeSeries['dy'] = (traj2.dict['Y'] - traj1.dict['Y'])/PTL.scale
            timeSeries['D2'] = (timeSeries['dx']**2 +  timeSeries['dy']**2)**0.5

            timeSeries['dz'] = (traj2.dict['Zr']*traj2.depthoStep - traj1.dict['Zr']*traj1.depthoStep)/1000
            timeSeries['dz'] *= PTL.OptCorrFactor
            timeSeries['D3'] = (timeSeries['D2']**2 +  timeSeries['dz']**2)**0.5

            #print('\n\n* timeSeries:\n')
            #print(timeSeries_DF[['T','B','F','dx','dy','dz','D2','D3']])
            print(gs.BLUE + 'OK!' + gs.NORMAL)


    #### 6. Compute forces
        print(gs.BLUE + 'Computing forces...' + gs.NORMAL)
        Tf = time.time()
        if PTL.NB == 2:
            print(gs.GREEN + '1 pair force computation' + gs.NORMAL)
            traj1 = PTL.listTrajectories[0]
            traj2 = PTL.listTrajectories[1]
            B0 = timeSeries['B']
            D3 = timeSeries['D3']
            dx = timeSeries['dx']
            F, dfLogF = PTL.computeForces(traj1, traj2, B0, D3, dx)
            # Main force computation function
            timeSeries['F'] = F

        print(gs.BLUE + 'OK! dT = {:.3f}'.format(time.time()-Tf) + gs.NORMAL)

            # Magnetization [A.m^-1]
            # M270
            # M = 0.74257*1.05*1600*(0.001991*B.^3+17.54*B.^2+153.4*B)./(B.^2+35.53*B+158.1)
            # M450
            # M = 1.05*1600*(0.001991*B.^3+17.54*B.^2+153.4*B)./(B.^2+35.53*B+158.1);


    #### 7. Export the results

        #### 7.1 - Save the tables !
        if PTL.NB == 2:
            timeSeries_DF = pd.DataFrame(timeSeries)
            timeSeriesFilePath = os.path.join(PathResultsDir, f[:-4] + '_PY.csv')
            timeSeries_DF.to_csv(timeSeriesFilePath, sep = ';', index=False)
            
    
    print(gs.BLUE + '\nTotal time:' + gs.NORMAL)
    print(gs.BLUE + str(time.time()-start) + gs.NORMAL)
    print(gs.BLUE + '\n' + gs.NORMAL)



# %% (2) Depthograph making classes & functions

# %%%% BeadDeptho

class BeadDeptho:
    def __init__(self, I, X0, Y0, S0, bestZ, scale, beadType, fileName):

        nz, ny, nx = I.shape[0], I.shape[1], I.shape[2]

        self.I = I
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.scale = scale
        self.X0 = X0
        self.Y0 = Y0
        self.S0 = S0
        self.XYm = np.zeros((self.nz, 2))
        self.XYm[S0-1, 0] = X0
        self.XYm[S0-1, 1] = Y0
        self.fileName = fileName

        self.beadType = beadType
        dictD0 = {'M450':4.5, 'M270':2.7}
        self.D0 = dictD0[beadType]
        self.I_cleanROI = np.array([])

        self.validBead = True
        self.iValid = -1

        self.bestZ = bestZ
        # self.validSlice = np.zeros(nz, dtype = bool)
        self.zFirst = 0
        self.zLast = nz
        self.validDepth = nz

        self.valid_v = True
        self.depthosDict = {}
        self.profileDict = {}
        self.ZfocusDict = {}



    def buildCleanROI(self, plot = 0):
        # Determine if the bead is to close to the edge on the max frame
        D0 = self.D0
        roughSize = np.floor(1.1*D0*self.scale)
        mx, Mx = np.min(self.X0 - 0.5*roughSize), np.max(self.X0 + 0.5*roughSize)
        my, My = np.min(self.Y0 - 0.5*roughSize), np.max(self.Y0 + 0.5*roughSize)
        testImageSize = mx > 0 and Mx < self.nx and my > 0 and My < self.ny

        # Aggregate the different validity test (for now only 1)
        validBead = testImageSize

        # If the bead is valid we can proceed
        self.validBead = validBead
        
        if not validBead:
            print('invalid image size')

        if validBead:
            for z in range(self.bestZ, -1, -1):
                if not z in self.S0:
                    break
            zFirst = z
            for z in range(self.bestZ, self.nz, +1):
                if not z in self.S0:
                    break
            zLast = z-1

            roughSize = int(np.floor(1.05*self.D0*self.scale))
            roughSize += 1 + roughSize%2
            roughCenter = int((roughSize+1)//2)

            cleanSize = ufun.getDepthoCleanSize(self.D0, self.scale)
            I_cleanROI = np.zeros([self.nz, cleanSize, cleanSize])

            # try:
            for i in range(zFirst, zLast):
                xmi, ymi = self.XYm[i,0], self.XYm[i,1]
                x1, y1, x2, y2, validBead = ufun.getROI(roughSize, xmi, ymi, self.nx, self.ny)
                if not validBead:
                    if y1 < 0 or y2 > self.ny:
                        self.valid_v = False


                xm1, ym1 = xmi-x1-0.5, ymi-y1-0.5
                I_roughRoi = self.I[i,y1:y2,x1:x2]

                translation = (xm1-roughCenter, ym1-roughCenter)

                tform = transform.EuclideanTransform(rotation=0, translation = translation)

                I_tmp = transform.warp(I_roughRoi, tform, order = 1, preserve_range = True)

                I_cleanROI[i] = np.copy(I_tmp[roughCenter-cleanSize//2:roughCenter+cleanSize//2+1,\
                                              roughCenter-cleanSize//2:roughCenter+cleanSize//2+1])

            if not self.valid_v:
                print('invalid vertical slice')
                self.validBead = False

            else:
                self.zFirst = zFirst
                self.zLast = zLast
                self.validDepth = zLast-zFirst
                self.I_cleanROI = I_cleanROI.astype(np.uint16)
                
            if self.validDepth < self.nz * (2/3):
                print('invalid depth')
                print(self.validDepth, self.nz * (2/3), self.nz)
                self.validBead = False

            # VISUALISE
            if plot >= 2:
                self.plotROI()
                # for i in range(zFirst, zLast, 50):
                #     self.plotROI(i)

            # except:
            #     print('Error for the file: ' + self.fileName)


    # def buildCleanROI(self, plot = 0):
    #     # Determine if the bead is to close to the edge on the max frame
    #     D0 = self.D0
    #     roughSize = np.floor(1.1*D0*self.scale)
    #     mx, Mx = np.min(self.X0 - 0.5*roughSize), np.max(self.X0 + 0.5*roughSize)
    #     my, My = np.min(self.Y0 - 0.5*roughSize), np.max(self.Y0 + 0.5*roughSize)
    #     testImageSize = mx > 0 and Mx < self.nx and my > 0 and My < self.ny

    #     # Aggregate the different validity test (for now only 1)
    #     validBead = testImageSize

    #     # If the bead is valid we can proceed
    #     self.validBead = validBead

    #     if validBead:
    #         for z in range(self.bestZ, -1, -1):
    #             if not z in self.S0:
    #                 break
    #         zFirst = z
    #         for z in range(self.bestZ, self.nz, +1):
    #             if not z in self.S0:
    #                 break
    #         zLast = z-1

    #         roughSize = int(np.floor(1.05*self.D0*self.scale))
    #         roughSize += 1 + roughSize%2
    #         roughCenter = int((roughSize+1)//2)

    #         cleanSize = ufun.getDepthoCleanSize(self.D0, self.scale)
    #         I_cleanROI = np.zeros([self.nz, cleanSize, cleanSize])

    #         # try:
    #         for i in range(zFirst, zLast):
    #             xmi, ymi = self.XYm[i,0], self.XYm[i,1]
    #             x1, y1, x2, y2, validBead = ufun.getROI(roughSize, xmi, ymi, self.nx, self.ny)
    #             if not validBead:
    #                 if y1 < 0 or y2 > self.ny:
    #                     self.valid_v = False


    #             xm1, ym1 = xmi-x1, ymi-y1
    #             I_roughRoi = self.I[i,y1:y2,x1:x2]

    #             translation = (xm1-roughCenter, ym1-roughCenter)

    #             tform = transform.EuclideanTransform(rotation=0, translation = translation)

    #             I_tmp = transform.warp(I_roughRoi, tform, order = 1, preserve_range = True)

    #             I_cleanROI[i] = np.copy(I_tmp[roughCenter-cleanSize//2:roughCenter+cleanSize//2+1,\
    #                                           roughCenter-cleanSize//2:roughCenter+cleanSize//2+1])

    #         if not self.valid_v:
    #             self.validBead = False

    #         else:
    #             self.zFirst = zFirst
    #             self.zLast = zLast
    #             self.validDepth = zLast-zFirst
    #             self.I_cleanROI = I_cleanROI.astype(np.uint16)
                
    #         if self.validDepth < self.nz * (2/3):
    #             self.validBead = False

    #         # VISUALISE
    #         if plot >= 2:
    #             self.plotROI()
    #             # for i in range(zFirst, zLast, 50):
    #             #     self.plotROI(i)

    #         # except:
    #         #     print('Error for the file: ' + self.fileName)


    def buildDeptho(self, nbPixToAvg = 5, interpolationFactor = 5):
        side_ROI = self.I_cleanROI.shape[1]
        mid_ROI = side_ROI//2
        deptho_raw = np.zeros([self.nz, side_ROI], dtype = np.float64)
        deptho_interp = np.zeros([self.nz, side_ROI*interpolationFactor], dtype = np.float64)

        if self.validBead:
            for z in range(self.zFirst, self.zLast):
                # templine = side_ROI
                deptho_raw[z] = self.I_cleanROI[z,:,mid_ROI] * (1/nbPixToAvg) # nbPixToAvg has to be an odd number
                for i in range(1, 1 + nbPixToAvg//2): # nbPixToAvg has to be an odd number
                    deptho_raw[z] += self.I_cleanROI[z,:,mid_ROI - i] * (1/nbPixToAvg)
                    deptho_raw[z] += self.I_cleanROI[z,:,mid_ROI + i] * (1/nbPixToAvg)
            deptho_raw = deptho_raw.astype(np.uint16)
            self.depthosDict['deptho_raw'] = deptho_raw


            for z in range(self.zFirst, self.zLast):
                xmin, xmax = mid_ROI - nbPixToAvg//2, mid_ROI + nbPixToAvg//2 + 1 # width = nbPixToAvg
                vals = self.I_cleanROI[z, :, xmin:xmax] # width = nbPixToAvg
    
                vals_new = ufun.resize_2Dinterp(vals, fx=1, fy=interpolationFactor) # height = cleanSize * interpolationFactor
                
                deptho_interp[z] = vals_new[:,nbPixToAvg//2] * (1/nbPixToAvg)
                for i in range(1, 1 + nbPixToAvg//2):
                    deptho_interp[z] += vals_new[:,nbPixToAvg//2-i] * (1/nbPixToAvg)
                    deptho_interp[z] += vals_new[:,nbPixToAvg//2+i] * (1/nbPixToAvg)

            deptho_interp = deptho_interp.astype(np.uint16)
            self.depthosDict['deptho_interp'] = deptho_interp

        # Raw profiles
        Z = np.array([z for z in range(self.I_cleanROI.shape[0])])
        mid_ROI_raw = deptho_raw.shape[1]//2
        intensity_raw = np.array([np.sum(deptho_raw[z,mid_ROI_raw-2:mid_ROI_raw+3])/5 
                                  for z in range(deptho_raw.shape[0])]).astype(np.float64)
        mid_ROI_interp = deptho_interp.shape[1]//2
        intensity_interp = np.array([np.sum(deptho_interp[z,mid_ROI_interp-2:mid_ROI_interp+3])/5 
                                     for z in range(deptho_interp.shape[0])]).astype(np.float64)
        
        self.profileDict['intensity_raw'] = intensity_raw
        self.profileDict['intensity_interp'] = intensity_interp
            
    
    def computeFocus(self, plot = 0):
        if plot >= 1:
            fig, axes = plt.subplots(1, 2)
            fig.suptitle('Focus')
        
        # raw
        nz, nx = self.depthosDict['deptho_raw'].shape
        
        Zm_intensity_raw = np.argmax(self.profileDict['intensity_raw'])
        intensity_raw_smooth = savgol_filter(self.profileDict['intensity_raw'], 101, 5)
        Zm_intensity_raw_smooth = np.argmax(intensity_raw_smooth)
        self.ZfocusDict['Zm_intensity_raw'] = Zm_intensity_raw_smooth
        
        STD_raw = np.std(self.depthosDict['deptho_raw'], axis = 1)
        STD_raw_smooth = savgol_filter(STD_raw, 101, 5)
        Zm_STD_raw = np.argmax(STD_raw_smooth)
        self.ZfocusDict['Zm_STD_raw'] = Zm_STD_raw
        
        if plot >= 1:
            Z = np.arange(nz)
            ax1 = axes[0]
            ax2 = ax1.twinx()
            
            ax1.plot(Z, self.profileDict['intensity_raw'], label='intensity_raw', color='k', ls='-')
            ax1.plot(Z, intensity_raw_smooth, label='intensity_raw_smooth', color='cyan', ls='-')
            ax1.axvline(Zm_intensity_raw_smooth, label='Zm_intensity_raw', color='b', ls='--')
            
            ax2.plot([], [], label='intensity_raw', color='k', ls='-')
            ax2.plot([], [], label='intensity_raw_smooth', color='cyan', ls='-')
            ax2.plot([], [], label='Zm_intensity_raw = {:.3f}'.format(Zm_intensity_raw), color='b', ls='--')
            ax2.plot(Z, STD_raw, label='STD_raw', color='orange', ls='-')
            ax2.plot(Z, STD_raw_smooth, label='STD_raw_smooth', color='red', ls='-')
            ax2.axvline(Zm_STD_raw, label='Zm_STD_raw = {:.3f}'.format(Zm_STD_raw), color='darkred', ls='--')
            
            ax2.legend()
        
        # interp
        nz, nx = self.depthosDict['deptho_interp'].shape
        
        Zm_intensity_interp = np.argmax(self.profileDict['intensity_interp'])
        intensity_interp_smooth = savgol_filter(self.profileDict['intensity_interp'], 101, 5)
        Zm_intensity_interp_smooth = np.argmax(intensity_interp_smooth)
        self.ZfocusDict['Zm_intensity_interp'] = Zm_intensity_interp_smooth
        
        STD_interp = np.std(self.depthosDict['deptho_interp'], axis = 1)
        STD_interp_smooth = savgol_filter(STD_interp, 101, 5)
        Zm_STD_interp = np.argmax(STD_interp_smooth)
        self.ZfocusDict['Zm_STD_interp'] = Zm_STD_interp        
        
        if plot >= 1:
            Z = np.arange(nz)
            ax1 = axes[1]
            ax2 = ax1.twinx()
            
            ax1.plot(Z, self.profileDict['intensity_interp'], label='intensity_interp', color='k', ls='-')           
            ax1.plot(Z, intensity_interp_smooth, label='intensity_interp_smooth', color='cyan', ls='-')
            ax1.axvline(Zm_intensity_interp_smooth, label='Zm_intensity_raw', color='b', ls='--')
            
            ax2.plot([], [], label='intensity_interp', color='k', ls='-')
            ax2.plot([], [], label='intensity_interp_smooth', color='cyan', ls='-')
            ax2.plot([], [], label='Zm_intensity_interp = {:.3f}'.format(Zm_intensity_interp), color='b', ls='--')
            ax2.plot(Z, STD_interp, label='STD_interp', color='orange', ls='-')
            ax2.plot(Z, STD_interp_smooth, label='STD_interp_smooth', color='red', ls='-')
            ax2.axvline(Zm_STD_interp, label='Zm_STD_interp = {:.3f}'.format(Zm_STD_interp), color='darkred', ls='--')
            
            ax2.legend()
            plt.tight_layout()
            plt.show()


    def saveBeadDeptho(self, path, ID, step, bestDetphoType = 'interp', bestFocusType = 'STD_interp'):
        supDataDir = ID + '_supData'
        supDataDirPath = os.path.join(path, supDataDir)
        if not os.path.exists(supDataDirPath):
            os.makedirs(supDataDirPath)

        cleanROIName = ID + '_cleanROI.tif'
        cleanROIPath = os.path.join(path, cleanROIName)
        io.imsave(cleanROIPath, self.I_cleanROI, check_contrast=False)

        profilesRaw_keys = ['intensity_raw', 'intensity_interp'] #, 'intensity_tot']
        profileDictRaw = {k: self.profileDict[k] for k in profilesRaw_keys}
        profileDictRaw_df = pd.DataFrame(profileDictRaw)
        profileDictRaw_df.to_csv(os.path.join(supDataDirPath, 'profiles_raw.csv'))

        ZfocusDict_df = pd.DataFrame(self.ZfocusDict, index = [1])
        ZfocusDict_df.to_csv(os.path.join(supDataDirPath, 'Zfoci.csv'))

        bestFocus = self.ZfocusDict['Zm_' + bestFocusType]
        metadataPath = os.path.join(path, ID + '_Metadata.csv')
        with open(metadataPath, 'w') as f:
            f.write('step;bestFocus')
#             for k in self.ZfocusDict.keys():
#                 f.write(';')
#                 f.write(k)
            f.write('\n')
            f.write(str(step) + ';' + str(bestFocus))
#             for k in self.ZfocusDict.keys():
#                 f.write(';')
#                 f.write(str(self.ZfocusDict[k]))

        depthoPath = os.path.join(path, ID + '_deptho.tif')
        bestDeptho = self.depthosDict['deptho_' + bestDetphoType]
        io.imsave(depthoPath, bestDeptho)



# Plot functions

    def plotXYm(self):
        fig, ax = plt.subplots(1,1)
        pStart, pStop = np.percentile(self.I[self.z_max], (1, 99))
        ax.imshow(self.I[self.z_max], cmap = 'gray', vmin = pStart, vmax = pStop)
        ax.plot(self.XYm[self.zFirst:self.zLast, 0], self.XYm[self.zFirst:self.zLast, 1],'r-')
        fig.show()

    def plotROI(self, i = 'auto'):
        if i == 'auto':
            i = self.bestZ

        fig, ax = plt.subplots(1,2, figsize = (16,4))     

        xm, ym = np.mean(self.XYm[self.zFirst:self.zLast,0]),  np.mean(self.XYm[self.zFirst:self.zLast,1])
        ROIsize_x = self.D0*1.25*self.scale + (max(self.XYm[self.zFirst:self.zLast, 0]) - min(self.XYm[self.zFirst:self.zLast, 0]))
        ROIsize_y = self.D0*1.25*self.scale + (max(self.XYm[self.zFirst:self.zLast, 1]) - min(self.XYm[self.zFirst:self.zLast, 1]))
        x1_ROI, y1_ROI, x2_ROI, y2_ROI = int(xm - ROIsize_x//2), int(ym - ROIsize_y//2), int(xm + ROIsize_x//2), int(ym + ROIsize_y//2)

        pStart, pStop = np.percentile(self.I[i], (1, 99))
        ax[0].imshow(self.I[i], cmap = 'gray', vmin = pStart, vmax = pStop)
        ax[0].plot([x1_ROI,x1_ROI], [y1_ROI,y2_ROI], 'c--')
        ax[0].plot([x1_ROI,x2_ROI], [y2_ROI,y2_ROI], 'c--')
        ax[0].plot([x2_ROI,x2_ROI], [y1_ROI,y2_ROI], 'c--')
        ax[0].plot([x1_ROI,x2_ROI], [y1_ROI,y1_ROI], 'c--')

        I_ROI = self.I[i,y1_ROI:y2_ROI,x1_ROI:x2_ROI]
        pStart, pStop = np.percentile(I_ROI, (1, 99))
        ax[1].imshow(I_ROI, cmap = 'gray', vmin = pStart, vmax = pStop)
        ax[1].plot(self.XYm[self.zFirst:self.zLast,0]-x1_ROI, self.XYm[self.zFirst:self.zLast,1]-y1_ROI, 'r-', lw=0.75)
        ax[1].plot(self.XYm[i,0]-x1_ROI, self.XYm[i,1]-y1_ROI, 'b+', lw=0.75)
        
        fig.show()


    def plotDeptho(self, d = '_interp'):
        fig, ax = plt.subplots(1,1, figsize = (4, 6))
        D = self.depthosDict['deptho_' + d]
        z_focus = self.ZfocusDict['Zm_' + d + '_interp']
        ny, nx = D.shape[0], D.shape[1]
        pStart, pStop = np.percentile(D, (1, 99))
        pStop = pStop + 0.3 * (pStop-pStart)
        ax.imshow(D, cmap='plasma', vmin = pStart, vmax = pStop)
        ax.plot([0, nx], [self.zFirst, self.zFirst], 'r--')
        ax.text(nx//2, self.zFirst - 10, str(self.zFirst), c = 'r')
        ax.plot([0, nx], [self.zLast, self.zLast], 'r--')
        ax.text(nx//2, self.zLast - 10, str(self.zLast), c = 'r')
        ax.plot([nx//2], [z_focus], 'c+')
        ax.text(nx//2, z_focus - 10, str(z_focus), c = 'c')
        fig.suptitle('File ' + self.fileName + ' - Bead ' + str(self.iValid))
        fig.show()


# %%%% depthoMaker


def depthoMaker(dictPaths, dictConstants):
        
    PathZStacks = dictPaths['PathZStacks']
    PathDeptho = dictPaths['PathDeptho']
    NameDeptho = dictPaths['NameDeptho']
    
    bead_type = dictConstants['bead type']
    scale = dictConstants['scale pixel per um']
    step = dictConstants['step']
    
    #### SETTINGS
    bestDetphoType = 'interp'
    bestFocusType = 'intensity_interp'
    
    rawFileList = os.listdir(PathZStacks)
    listFileNames = [f[:-4] for f in rawFileList if (os.path.isfile(os.path.join(PathZStacks, f)) and f.endswith(".tif"))]
    L = []

    for f in listFileNames:
        test1 = ((f + '_Results.txt') in os.listdir(PathZStacks))
        valid = test1 and True
        if valid:
            L.append(f)

    listFileNames = L

    listBD = []
#     dictBD = {}

#     print(listFileNames)
    for f in listFileNames:
        filePath = os.path.join(PathZStacks, f)
        I = io.imread(filePath + '.tif')
        resDf = pd.read_csv((filePath + '_Results.txt'), sep = '\t').drop(columns = [' '])
        # Area,StdDev,XM,YM,Slice
        X0 = resDf['XM'].values
        Y0 = resDf['YM'].values
        S0 = resDf['Slice'].values
        bestZ = S0[np.argmax(resDf['StdDev'].values)] - 1 # The index of the image with the highest Std
        # This image will be more or less the one with the brightest spot

        # Create the BeadDeptho object
        BD = BeadDeptho(I, X0, Y0, S0, bestZ, scale, bead_type, f)

        # Creation of the clean ROI where the center of mass is always perfectly centered.
        BD.buildCleanROI()

        # If the bead was not acceptable (for instance too close to the edge of the image)
        # then BD.validBead will be False
        if not BD.validBead:
            print(gs.RED + 'Not acceptable file: ' + f + gs.NORMAL)

        # Else, we can proceed.
        else:
            # Creation of the z profiles
            BD.buildDeptho()
            BD.computeFocus()
            listBD.append(BD)
            print(gs.BLUE + 'Job done for the file: ' + f + gs.NORMAL)
            
        i = 1
        for BD in listBD:
            subFileSavePath = os.path.join(PathDeptho, 'Intermediate_Py', NameDeptho)
            BD.saveBeadDeptho(subFileSavePath, f, step = step, bestDetphoType = bestDetphoType, bestFocusType = bestFocusType)
            i += 1

    maxAboveZm, maxBelowZm = 0, 0
    for BD in listBD:
        Zm = int(np.round(BD.ZfocusDict['Zm_' + bestFocusType]))
        if Zm - BD.zFirst > maxAboveZm:
            maxAboveZm = Zm - BD.zFirst
        if BD.zLast - Zm > maxBelowZm:
            maxBelowZm = BD.zLast - Zm
    maxAboveZm, maxBelowZm = int(maxAboveZm), int(maxBelowZm)
    Zfocus = maxAboveZm
    depthoWidth = listBD[0].depthosDict['deptho_' + bestDetphoType].shape[1]
    depthoHeight = maxAboveZm + maxBelowZm
    finalDeptho = np.zeros([depthoHeight, depthoWidth], dtype = np.float64)

    for z in range(1, maxAboveZm+1):
        count = 0
        for BD in listBD:
            Zm = int(np.round(BD.ZfocusDict['Zm_' + bestFocusType]))
            currentDeptho = BD.depthosDict['deptho_' + bestDetphoType]
            if Zm-z >= 0 and np.sum(currentDeptho[Zm-z,:] != 0):
                count += 1
        for BD in listBD:
            Zm = int(np.round(BD.ZfocusDict['Zm_' + bestFocusType]))
            currentDeptho = BD.depthosDict['deptho_' + bestDetphoType]
            if Zm-z >= 0 and np.sum(currentDeptho[Zm-z,:] != 0):
                finalDeptho[Zfocus-z,:] += currentDeptho[Zm-z,:]/count

    for z in range(0, maxBelowZm):
        count = 0
        for BD in listBD:
            Zm = int(np.round(BD.ZfocusDict['Zm_' + bestFocusType]))
            currentDeptho = BD.depthosDict['deptho_interp']
#             print(currentDeptho.shape)
            if Zm+z >= 0 and Zm+z < currentDeptho.shape[0] and np.sum(currentDeptho[Zm+z,:] != 0):
                count += 1
        for BD in listBD:
            Zm = int(np.round(BD.ZfocusDict['Zm_' + bestFocusType]))
            currentDeptho = BD.depthosDict['deptho_' + bestDetphoType]
            if Zm+z >= 0 and Zm+z < currentDeptho.shape[0] and np.sum(currentDeptho[Zm+z,:] != 0):
                finalDeptho[Zfocus+z,:] += currentDeptho[Zm+z,:]/count

    # print(Zm, maxAboveZm, maxBelowZm)
    finalDeptho = finalDeptho.astype(np.uint16)

    fig, ax = plt.subplots(1,1)
    ax.imshow(finalDeptho)

    fig.suptitle(NameDeptho)
    fig.show()

    depthoSavePath = os.path.join(PathDeptho, NameDeptho)
    io.imsave(depthoSavePath, finalDeptho)
    metadataPath = os.path.join(PathDeptho, NameDeptho[:-4] + '_Metadata.csv')
    with open(metadataPath, 'w') as f:
        f.write('step;focus')
        f.write('\n')
        f.write(str(step) + ';' + str(Zfocus))

    print(gs.GREEN + 'ok' + gs.NORMAL)


# Finished !

