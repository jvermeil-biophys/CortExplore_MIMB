# -*- coding: utf-8 -*-
"""
UtilityFunctions.py - contains all kind of small functions used by CortExplore programs, 
to be imported with "import UtilityFunctions as ufun" and call with "ufun.my_function".
Joseph Vermeil, Anumita Jawahar, 2022

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
import statsmodels.api as sm
import matplotlib.pyplot as plt

import os
import re
import shutil
import numbers
import pyautogui

from skimage import io, filters, exposure, util, color
from matplotlib.gridspec import GridSpec
from PyQt5 import QtWidgets as Qtw
from collections.abc import Collection
from copy import deepcopy

#### Local Imports

import sys
# import CortexPaths as cp

import GraphicStyles as gs
# import GlobalConstants as gc

# 2. Pandas settings
pd.set_option('mode.chained_assignment', None)


# 4. Other settings
# These regex are used to correct the stupid date conversions done by Excel
# When given a date yy-mm-dd, excel will convert in yy/mm/20dd or yy/mm/19dd if yy >= 30
dateFormatExcel = re.compile(r'[1-2]\d{1}/\d{2}/(?:19|20)\d{2}') # matches X#/##/YY## where X in {1, 2} and YY in {19, 20}
dateFormatExcel2 = re.compile(r'[1-2]\d-\d{2}-(?:19|20)\d{2}') # matches X#-##-YY## where X in {1, 2} and YY in {19, 20}
dateFormatOk = re.compile(r'\d{2}-\d{2}-\d{2}') # Correct format yy-mm-dd we use


# %% (1) Utility functions


# %%% Image management - nomeclature and creating stacks

def AllMMTriplets2Stack(DirExt, DirSave, prefix, channel):
    """
    Used for metamorph created files.
    Metamoprh does not save images in stacks but individual triplets. These individual triplets take time
    to open in FIJI.
    This function takes images of a sepcific channel and creates .tif stacks from them.
       
    """
    
    allCells = os.listdir(DirExt)
    excludedCells = []
    for currentCell in allCells:
        dirPath = os.path.join(DirExt, currentCell)
        allFiles = os.listdir(dirPath)
        date = findInfosInFileName(currentCell, 'date')
        
        # date = date.replace('-', '.')
        filename = currentCell+'_'+channel

        try:
            os.mkdir(DirSave+'/'+currentCell)
        except:
            pass
        
        allFiles = [dirPath+'/'+string for string in allFiles if 'thumb' not in string and '.TIF' in string and channel in string]
        #+4 at the end corrosponds to the '_t' part to sort the array well
        limiter = len(dirPath)+len(prefix)+len(channel)+4
        
        try:
            allFiles.sort(key=lambda x: int(x[limiter:-4]))
        except:
            print('Error in sorting files')
        
        try:
            ic = io.ImageCollection(allFiles, conserve_memory = True)
            stack = io.concatenate_images(ic)
            io.imsave(DirSave+'/'+currentCell+'/'+filename+'.tif', stack)
        except:
            excludedCells.append(currentCell)
            print(gs.ORANGE + "Unknown error in saving "+currentCell + gs.NORMAL)
            
    return excludedCells
        

def renamePrefix(DirExt, currentCell, newPrefix):
    """
    Used for metamorph created files.
    Metamorph creates a new folder for each timelapse, within which all images contain a predefined 
    'prefix' and 'channel' name which can differ between microscopes. Eg.: 'w1TIRF_DIC' or 'w2TIRF_561'
    
    If you forget to create a new folder for a new timelapse, Metamorph automatically changes the prefix
    to distinguish between the old and new timelapse triplets. This can get annoying when it comes to processing 
    many cells.
    
    This function allows you to rename the prefix of all individual triplets in a specific folder. 
    
    """
    
    path = os.path.join(DirExt, currentCell)
    allImages = os.listdir(path)
    for i in allImages:
        if i.endswith('.TIF'):
            split = i.split('_')
            if split[0] != newPrefix:
                split[0] = newPrefix
                newName = '_'.join(split)
                try:
                    os.rename(os.path.join(path,i), os.path.join(path, newName))
                except:
                    print(currentCell)
                    print(gs.ORANGE + "Error! There may be other files with the new prefix you can trying to incorporate" + gs.NORMAL)
        
        if i.endswith('.nd'):
            newName = newPrefix+'.nd'
            try:
                os.rename(os.path.join(path,i), os.path.join(path, newName))
            except:
                print(currentCell)
                print(gs.YELLOW + "Error! There may be other .nd files with the new prefix you can trying to incorporate" + gs.NORMAL)
            
# %%% Data management




def findFirstAbsTimeOfDate(src_path, date, suffix = '.csv'):
    """
    Given a directory containing time series (src_path), a date (date), 
    and a suffix (suffix='.csv'), goes through all the files which name contains 
    date and ends with suffix, and consider all the first elements of the columns named 'Tabs'.
    Return the min of these elements.
    
    Should be called with src_path = cp.DirDataTimeseries, date = chosen_date, suffix = '.csv',
    'chosen_date' being the date of an experiment in the format yy-mm-dd.
    """
    files = os.listdir(src_path)
    selected_files = []
    for f in files:
        if date in f and f.endswith(suffix):
            selected_files.append(f)
    dictOrd = {}
    listOrd = []
    for f in selected_files:
        ordVal = findInfosInFileName(f, 'ordinalValue')
        dictOrd[ordVal] = f
        listOrd.append(int(ordVal))
    try:
        minOrdVal = str(np.min(listOrd))
        first_f = dictOrd[minOrdVal]
        first_f_path = os.path.join(src_path, first_f)
        df = pd.read_csv(first_f_path, sep = ';', usecols=['Tabs'], nrows=1)
        date_T0 = df['Tabs'].values[0]
        
    except:
        date_T0 = np.nan
    
    return(date_T0)
        


def removeColumnsDuplicate(df):
    cols = df.columns.values
    for c in cols:
        if c.endswith('_x'):
            df = df.rename(columns={c: c[:-2]})
        elif c.endswith('_y'):
            df = df.drop(columns=[c])
    return(df)


def findActivation(fieldDf):
    maxZidx = fieldDf['Z'].argmax() #Finding the index of the max Z
    maxZ = fieldDf['Z'][maxZidx] #To check if the value is correct
    return(maxZidx, maxZ)   



def findInfosInFileName(f, infoType):
    """
    Return a given type of info from a file name.
    Inputs : f (str), the file name.
             infoType (str), the type of info wanted.
             
             infoType can be equal to : 
                 
             * 'M', 'P', 'C' -> will return the number of manip (M), well (P), or cell (C) in a cellID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'C', the function will return 8.
             
             * 'manipID'     -> will return the full manip ID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'manipID', the function will return '21-01-18_M2'.
             
             * 'cellID'     -> will return the full cell ID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'cellID', the function will return '21-01-18_M2_P1_C8'.
             
             * 'substrate'  -> will return the string describing the disc used for cell adhesion.
             ex : if f = '21-01-18_M2_P1_C8_disc15um.tif' and infoType = 'substrate', the function will return 'disc15um'.
             
             * 'ordinalValue'  -> will return a value that can be used to order the cells. It is equal to M*1e6 + P*1e3 + C
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'ordinalValue', the function will return "2'001'008".
    """
    infoString = ''
    try:
        if infoType in ['M', 'P', 'C']:
            acceptedChar = [str(i) for i in range(10)] + ['-']
            string = '_' + infoType
            iStart = re.search(string, f).end()
            i = iStart
            infoString = '' + f[i]
            while f[i+1] in acceptedChar and i < len(f)-1:
                i += 1
                infoString += f[i]
                
        elif infoType in ['M_float', 'P_float', 'C_float']:
            acceptedChar = [str(i) for i in range(10)] + ['-']
            string = '_' + infoType[0]
            iStart = re.search(string, f).end()
            i = iStart
            infoString = '' + f[i]
            while f[i+1] in acceptedChar and i < len(f)-1:
                i += 1
                infoString += f[i]
            infoString = infoString.replace('-', '.')
            infoString = float(infoString)
                
        elif infoType == 'date':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            infoString = date
        
        elif infoType == 'manipID':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            manip = 'M' + findInfosInFileName(f, 'M')
            infoString = date + '_' + manip
            
        elif infoType == 'cellName':
            infoString = 'M' + findInfosInFileName(f, 'M') + \
                         '_' + 'P' + findInfosInFileName(f, 'P') + \
                         '_' + 'C' + findInfosInFileName(f, 'C')
            
        elif infoType == 'cellID':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            infoString = date + '_' + 'M' + findInfosInFileName(f, 'M') + \
                                '_' + 'P' + findInfosInFileName(f, 'P') + \
                                '_' + 'C' + findInfosInFileName(f, 'C')
                                
        elif infoType == 'substrate':
            try:
                pos = re.search(r"disc[\d]*um", f)
                infoString = f[pos.start():pos.end()]
            except:
                infoString = ''
                
        elif infoType == 'ordinalValue':
            M, P, C = findInfosInFileName(f, 'M'), findInfosInFileName(f, 'P'), findInfosInFileName(f, 'C')
            L = [M, P, C]
            for i in range(len(L)):
                s = L[i]
                if '-' in s:
                    s = s.replace('-', '.')
                    L[i] = s
            [M, P, C] = L
            ordVal = int(float(M)*1e9 + float(P)*1e6 + float(C)*1e3)
            infoString = str(ordVal)
            
    except:
        pass
                             
    return(infoString)


# def isFileOfInterest_V0(f, manips, wells, cells):
#     """
#     Determine if a file f correspond to the given criteria.
#     More precisely, return a boolean saying if the manip, well and cell number are in the given range.
#     f is a file name. Each of the fields 'manips', 'wells', 'cells' can be either a number, a list of numbers, or 'all'.
#     Example : if f = '21-01-18_M2_P1_C8.tif'
#     * manips = 'all', wells = 'all', cells = 'all' -> the function return True.
#     * manips = 1, wells = 'all', cells = 'all' -> the function return False.
#     * manips = [1, 2], wells = 'all', cells = 'all' -> the function return True.
#     * manips = [1, 2], wells = 2, cells = 'all' -> the function return False.
#     * manips = [1, 2], wells = 1, cells = [5, 6, 7, 8] -> the function return True.
#     Note : if manips = 'all', the code will consider that wells = 'all', cells = 'all'.
#            if wells = 'all', the code will consider that cells = 'all'.
#            This means you can add filters only in this order : manips > wells > cells.
#     """
#     test = False
#     if f.endswith(".tif"):
#         if manips == 'all':
#             test = True
#         else:
#             try:
#                 manips_str = [str(i) for i in manips]
#             except:
#                 manips_str = [str(manips)]
#             infoM = findInfosInFileName(f, 'M')
#             if infoM in manips_str:
#                 if wells == 'all':
#                     test = True
#                 else:
#                     try:
#                         wells_str = [str(i) for i in wells]
#                     except:
#                         wells_str = [str(wells)]
#                     infoP = findInfosInFileName(f, 'P')
#                     if infoP in wells_str:
#                         if cells == 'all':
#                             test = True
#                         else:
#                             try:
#                                 cells_str = [str(i) for i in cells]
#                             except:
#                                 cells_str = [str(cells)]
#                             infoC = findInfosInFileName(f, 'C')
#                             if infoC in cells_str:
#                                 test = True
#     return(test)



def isFileOfInterest(f, manips, wells, cells):
    """
    Determine if a file f correspond to the given criteria.
    More precisely, return a boolean saying if the manip, well and cell number are in the given range.
    f is a file name. Each of the fields 'manips', 'wells', 'cells' can be either a number, a list of numbers, or 'all'.
    Example : if f = '21-01-18_M2_P1_C8.tif'
    * manips = 'all', wells = 'all', cells = 'all' -> the function return True.
    * manips = 1, wells = 'all', cells = 'all' -> the function return False.
    * manips = [1, 2], wells = 'all', cells = 'all' -> the function return True.
    * manips = [1, 2], wells = 2, cells = 'all' -> the function return False.
    * manips = [1, 2], wells = 1, cells = [5, 6, 7, 8] -> the function return True.
    Example2 : if f = '21-01-18_M2_P1_C8-1.tif'
    * manips = [1, 2], wells = 1, cells = [5, 6, 7, 8] -> the function return False.
    * manips = [1, 2], wells = 1, cells = [5, 6, 7, '8-1'] -> the function return True.
    Note : if manips = 'all', the code will consider that wells = 'all', cells = 'all'.
           if wells = 'all', the code will consider that cells = 'all'.
           This means you can add filters only in this order : manips > wells > cells.
    """
    test = False
    testM, testP, testC = False, False, False
    
    listManips, listWells, listCells = toListOfStrings(manips), toListOfStrings(wells), toListOfStrings(cells)
    
    try:
        # fM = int(findInfosInFileName(f, 'M'))
        # fP = int(findInfosInFileName(f, 'P'))
        # fC = int(findInfosInFileName(f, 'C'))
        fM = (findInfosInFileName(f, 'M'))
        fP = (findInfosInFileName(f, 'P'))
        fC = (findInfosInFileName(f, 'C'))
    except:
        return(False)
        
    # print(fM, fP, fC)
    # print(manips, wells, cells)
    # print(toList(manips), toList(wells), toList(cells))
    
    if (manips == 'all') or (fM in listManips):
        testM = True
    if (wells == 'all') or (fP in listWells):
        testP = True
    if (cells == 'all') or (fC in listCells):
        testC = True
        
    if testM and testP and testC:
        test = True
    
    return(test)

def getDictAggMean(df):
    dictAggMean = {}
    for c in df.columns:
        # print(c)
        S = df[c].dropna()
        lenNotNan = S.size
        if lenNotNan == 0:
            dictAggMean[c] = 'first'
        else:
            S.infer_objects()
            if S.dtype == bool:
                dictAggMean[c] = np.nanmin
            else:
                # print(c)
                # print(pd.Series.all(S.apply(lambda x : isinstance(x, numbers.Number))))
                if pd.Series.all(S.apply(lambda x : isinstance(x, numbers.Number))):
                    dictAggMean[c] = np.nanmean
                else:
                    dictAggMean[c] = 'first'
                    
    if 'compNum' in dictAggMean.keys():
        dictAggMean['compNum'] = np.nanmax
                    
    return(dictAggMean)

# def getDictAggMean_V0(df):
#     dictAggMean = {}
#     for c in df.columns:
#     #         t = df[c].dtype
#     #         print(c, t)
#             try :
#                 if np.array_equal(df[c], df[c].astype(bool)):
#                     dictAggMean[c] = 'min'
#                 else:
#                     try:
#                         if not c.isnull().all():
#                             np.mean(df[c])
#                             dictAggMean[c] = 'mean'
#                     except:
#                         dictAggMean[c] = 'first'
#             except:
#                     dictAggMean[c] = 'first'
#     return(dictAggMean)



        
    
def updateDefaultSettingsDict(settingsDict, defaultSettingsDict):
    """
    Update defaultSettingDict with new values contained in settingDict.

    Parameters
    ----------
    settingDict : dict
        Contains new {SettingName (string) : SettingValue (object)} pairs.
    defaultSettingDict : dict
        Contains all default {SettingName (string) : SettingValue (object)} pairs.

    Returns
    -------
    newSettingDict : dict
        Contains all default {SettingName (string) : SettingValue (object)} pairs, 
        but updated where new setting values were inputed from settingDict.
        
    Example
    -------
    >>> defaultSettingsDict = {'A':1, 'B': 10, 'C': True}
    >>> settingsDict = {'A':1000, 'D': False}
    >>> newSettingsDict = updateDefaultSettingsDict(settingDict, 
                                                  defaultSettingDict)
    >>> newSettingsDict
    Out[1]: {'A': 1000, 'B': 10, 'C': True, 'D': False}
    
    >>> defaultSettingsDict = {'A':1, 'B': 10, 'C': True}
    >>> settingsDict = {} # We don't want to modify the default settings
    >>> newSettingsDict = updateDefaultSettingsDict(settingsDict, 
                                                    defaultSettingsDict)
    >>> newSettingsDict
    Out[2]: {'A':1, 'B': 10, 'C': True}
    
    IMPORTANT NOTE
    --------------
    I'M GUTTED !!!! This is exactly what the method dict1.update(dict2) does !!!
    At least it shows that my way of proceeding isn't totally weird.
    """
    
    newSettingsDict = deepcopy(defaultSettingsDict)
    for k in settingsDict.keys():
        newSettingsDict[k] = settingsDict[k]
    return(newSettingsDict)



def flattenPandasIndex(pandasIndex):
    """
    Flatten a multi-leveled pandas index.

    Parameters
    ----------
    pandasIndex : pandas MultiIndex
        Example: MultiIndex([('course', ''),('A', 'count'),('A', 'sum'),( 'coeff', 'sum')])

    Returns
    -------
    new_pandasIndex : list that can be reasigned as a flatten index.
        In the former example: new_pandasIndex = ['course', 'A_count', 'A_sum', 'coeff_sum']
        
    Example
    -------
    >>> data_agg.columns
    >>> Out: MultiIndex([('course',      ''),
    >>>                 (     'A', 'count'),
    >>>                 (     'A',   'sum'),
    >>>                 ( 'coeff',   'sum')],
    >>>                )
    >>> data_agg.columns = flattenPandasIndex(data_agg.columns)
    >>> data_agg.columns
    >>> Out: Index(['course', 'A_count', 'A_sum', 'coeff_sum'], dtype='object')

    """
    new_pandasIndex = []
    for idx in pandasIndex:
        new_idx = '_'.join(idx)
        while new_idx[-1] == '_':
            new_idx = new_idx[:-1]
        new_pandasIndex.append(new_idx)
    return(new_pandasIndex)


# %%% File manipulation

def copyFile(DirSrc, DirDst, filename):
    """
    Copy the file 'filename' from 'DirSrc' to 'DirDst'
    """
    PathSrc = os.path.join(DirSrc, filename)
    PathDst = os.path.join(DirDst, filename)
    shutil.copyfile(PathSrc, PathDst)
    
def copyFolder(DirSrc, DirDst, folderName):
    """
    Copy the folder 'folderName' from 'DirSrc' to 'DirDst' with all its contents
    """
    PathSrc = os.path.join(DirSrc, folderName)
    PathDst = os.path.join(DirDst, folderName)
    shutil.copytree(PathSrc, PathDst)
    
def copyFilesWithString(DirSrc, DirDst, stringInName):
    """
    Copy all the files from 'DirSrc' which names contains 'stringInName' to 'DirDst'
    """
    SrcFilesList = os.listdir(DirSrc)
    for SrcFile in SrcFilesList:
        if stringInName in SrcFile:
            copyFile(DirSrc, DirDst, SrcFile)
            
def containsFilesWithExt(Dir, ext):
    answer = False
    FilesList = os.listdir(Dir)
    for File in FilesList:
        if File.endswith(ext):
            answer = True
    return(answer)

def softMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %%% Stats

def get_R2(Ymeas, Ymodel):
    meanY = np.mean(Ymeas)
    meanYarray = meanY*np.ones(len(Ymeas))
    SST = np.sum((Ymeas-meanYarray)**2)
    SSE = np.sum((Ymodel-meanYarray)**2)
    if pd.isnull(SST) or pd.isnull(SSE) or SST == 0:
        R2 = np.nan
    else:
        R2 = SSE/SST
    return(R2)

def get_Chi2(Ymeas, Ymodel, dof, err):
    if dof <= 0:
        Chi2_dof = np.nan
    else:
        residuals = Ymeas-Ymodel
        Chi2 = np.sum((residuals/err)**2)
        Chi2_dof = Chi2/dof
    return(Chi2_dof)

# %%% Image processing

def getDepthoCleanSize(D, scale):
    """
    Function that looks stupid but is quite important ! It allows to standardise 
    across all other functions the way the depthograph width is computed.
    D here is the approximative size of the bead in microns, 4.5 for M450, 2.7 for M270.
    Scale is the pixel to microns ration of the objective.
    """
    cleanSize = int(np.floor(1*D*scale))
    cleanSize += 1 + cleanSize%2
    return(cleanSize)

def compute_cost_matrix(XY1,XY2):
    """
    Compute a custom cost matrix between two arrays of XY positions.
    Here the costs are simply the squared distance between each XY positions.
    Example : M[2,1] is the sqaured distance between XY1[2] and XY2[1], 
    which is ((XY2[1,1]-XY1[2,1])**2 + (XY2[1,0]-XY1[2,0])**2)
    """
    N1, N2 = XY1.shape[0],XY2.shape[0]
    M = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            M[i,j] = (np.sum((XY2[j,:] - XY1[i,:]) ** 2))
    return(M)

def ui2array(uixy):
    """
    Translate the output of the function plt.ginput() 
    (which are lists of tuples), in an XY array with this shape:
    XY = [[x0, y0], [x1, y1], [x2, y2], ...]
    So if you need the [x, y] of 1 specific point, call XY[i]
    If you need the list of all x coordinates, call XY[:, 0]
    """
    n = len(uixy)
    XY = np.zeros((n, 2))
    for i in range(n):
        XY[i,0], XY[i,1] = uixy[i][0], uixy[i][1]
    return(XY)

def getROI(roiSize, x0, y0, nx, ny):
    """
    Return coordinates of top left (x1, y1) and bottom right (x2, y2) corner of a ROI, 
    and a boolean validROI that says if the ROI exceed the limit of the image.
    Inputs : 
    - roiSize, the width of the (square) ROI.
    - x0, y0, the position of the central pixel.
    - nx, ny, the size of the image.
    Note : the ROI is done so that the final width (= height) 
    of the ROI will always be an odd number.
    """
    roiSize += roiSize%2
    x1 = int(np.floor(x0) - roiSize*0.5) - 1
    x2 = int(np.floor(x0) + roiSize*0.5)
    y1 = int(np.floor(y0) - roiSize*0.5) - 1
    y2 = int(np.floor(y0) + roiSize*0.5)
    if min([x1,nx-x2,y1,ny-y2]) < 0:
        validROI = False
    else:
        validROI = True
    return(x1, y1, x2, y2, validROI)



def squareDistance(M, V, normalize = False): # MUCH FASTER ! **Michael Scott Voice** VERRRRY GOODE
    """
    Compute a distance between two arrays of the same size, defined as such:
    D = integral of the squared difference between the two arrays.
    It is used to compute the best fit of a slice of a bead profile on the depthograph.
    This function speed is critical for the Z computation process because it is called so many times !
    What made that function faster is the absence of 'for' loops and the use of np.repeat().
    """
    #     top = time.time()
    n, m = M.shape[0], M.shape[1]
    # len(V) should be m
    if normalize:
        V = V/np.mean(V)
    V = np.array([V])
    MV = np.repeat(V, n, axis = 0) # Key trick for speed !
    if normalize:
        M = (M.T/np.mean(M, axis = 1).T).T
    R = np.sum((M-MV)**2, axis = 1)
#     print('DistanceCompTime')
#     print(time.time()-top)
    return(R)

def matchDists(listD, listStatus, Nup, NVox, direction):
    """
    This function transform the different distances curves computed for 
    a Nuplet of images to match their minima. By definition it is not used for singlets of images.
    In practice, it's a tedious and boring function.
    For a triplet of image, it will move the distance curve by NVox voxels to the left 
    for the first curve of a triplet, not move the second one, and move the third by NVox voxels to the right.
    The goal : align the 3 matching minima so that the sum of the three will have a clear global minimum.
    direction = 'upward' or 'downward' depending on how your triplet images are taken 
    (i.e. upward = consecutively towards the bright spot and downwards otherwise)
    """
    N = len(listStatus)
    offsets = np.array(listStatus) - np.ones(N) * (Nup//2 + 1)
    offsets = offsets.astype(int)
    listD2 = []
    if direction == 'upward':
        for i in range(N):
            if offsets[i] < 0:
                shift = abs(offsets[i])*NVox
                D = listD[i]
                fillVal = max(D)
                D2 = np.concatenate((D[shift:],fillVal*np.ones(shift))).astype(np.float64)
                listD2.append(D2)
            if offsets[i] == 0:
                D = listD[i].astype(np.float64)
                listD2.append(D)
            if offsets[i] > 0:
                shift = abs(offsets[i])*NVox
                D = listD[i]
                fillVal = max(D)
                D2 = np.concatenate((fillVal*np.ones(shift),D[:-shift])).astype(np.float64)
                listD2.append(D2)
    elif direction == 'downward':
        for i in range(N):
            if offsets[i] > 0:
                shift = abs(offsets[i])*NVox
                D = listD[i]
                fillVal = max(D)
                D2 = np.concatenate((D[shift:],fillVal*np.ones(shift))).astype(np.float64)
                listD2.append(D2)
            if offsets[i] == 0:
                D = listD[i].astype(np.float64)
                listD2.append(D)
            if offsets[i] < 0:
                shift = abs(offsets[i])*NVox
                D = listD[i]
                fillVal = max(D)
                D2 = np.concatenate((fillVal*np.ones(shift),D[:-shift])).astype(np.float64)
                listD2.append(D2)
    return(np.array(listD2))


def uiThresholding(I, method = 'otsu', factorT = 0.8):
    """
    Interactive thresholding function to replace IJ.
    Compute an auto thresholding on a global 3D image with a method from this list:
    > 'otsu', 'max_entropy', (add the method you want ! here are the options : https://scikit-image.org/docs/stable/api/skimage.filters.html )
    Then display a figure for the user to assess the threshold fitness, and according to the user choice,
    confirm the threshold or recompute it with new parameters in a recursive way.
    """
    # 1. Compute the threshold
    nz = I.shape[0]
    if method == 'otsu':
        threshold = factorT*filters.threshold_otsu(I)
    elif method == 'max_entropy':
        bitDepth = util.dtype_limits(I)[1]+1
        I8 = util.img_as_ubyte(I)
        threshold = factorT*max_entropy_threshold(I8)*(bitDepth/2**8)
        
    # 2. Display images for the user to assess the fitness
    # New version of the plot
        nS = I.shape[0]
        loopSize = nS//4
        N = min(4, nS//loopSize)
        L_I_plot = [I[loopSize*2*k + 2] for k in range(N)]
        L_I_thresh = [I_plot > threshold for I_plot in L_I_plot]
        for i in range(N):
            I_plot = L_I_plot[i]
            I_thresh = L_I_thresh[i]
            I_plot = util.img_as_ubyte(I_plot)
            I_plot = color.gray2rgb(I_plot)
            pStart, pStop = np.percentile(I_plot, (1, 99))
            I_plot = exposure.rescale_intensity(I_plot, in_range=(pStart, pStop))
            red_multiplier = [255, 0, 0]
            I_plot[I_thresh] = red_multiplier
            L_I_plot[i] = I_plot
            
        I_thresh_all = I > threshold
        I_thresh_max = np.max(I_thresh_all, axis = 0)
        
        fig = plt.figure(tight_layout=True)
        gs = GridSpec(2, 4, figure=fig)
        ax = []
        for i in range(N):
            ax.append(fig.add_subplot(gs[i//2, i%2]))
            ax[-1].imshow(L_I_plot[i])
            ax[-1].set_title('Frame ' + str(loopSize*2*i + 2) + '/' + str(nS), fontsize = 8)
            ax[-1].axes.xaxis.set_ticks([])
            ax[-1].axes.yaxis.set_ticks([])
        ax.append(fig.add_subplot(gs[:, 2:]))
        ax[-1].imshow(I_thresh_max, cmap = 'gray')
        ax[-1].set_title('Max projection', fontsize = 10)
        ax[-1].axes.xaxis.set_ticks([])
        ax[-1].axes.yaxis.set_ticks([])
        fig.suptitle(str(threshold), fontsize = 12)
        fig.show()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50, 380, 1800, 650)
    
    # 3. Ask the question to the user
    QA = pyautogui.confirm(
                text='Is the threshold satisfying?',
                title='Confirm threshold', 
                buttons=['Yes', '10% Lower', '5% Lower', '1% Lower', '1% Higher', '5% Higher', '10% Higher'])
    plt.close(fig)
    
    # 4. Recall the same function with new parameters, or validate the threshold 
    # according to the user answer.
    increment = 0.1 * ('10%' in QA) + 0.05 * ('5%' in QA) + 0.01 * ('1%' in QA)
    if 'Lower' in QA:
        uiThresholding(method = method, factorT = factorT - increment)
    elif 'Higher' in QA:
        uiThresholding(method = method, factorT = factorT + increment)
    elif QA == 'Yes':
        threshold = threshold
    return(threshold)

def max_entropy(data):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    M. Emre Celebi
    06.15.2007
    Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
    2016-04-28: Adapted for Python 2.7 by Robert Metchev from Java source of MaxEntropy() in the Autothresholder plugin
    http://rsb.info.nih.gov/ij/plugins/download/AutoThresholder.java
    :param data: Sequence representing the histogram of the image
    :return threshold: Resulting maximum entropy threshold
    """

    # calculate CDF (cumulative density function)
    cdf = data.astype(np.float).cumsum()

    # find histogram's nonzero area
    valid_idx = np.nonzero(data)[0]
    first_bin = valid_idx[0]
    last_bin = valid_idx[-1]

    # initialize search for maximum
    max_ent, threshold = 0, 0

    for it in range(first_bin, last_bin + 1):
        # Background (dark)
        hist_range = data[:it + 1]
        hist_range = hist_range[hist_range != 0] / cdf[it]  # normalize within selected range & remove all 0 elements
        tot_ent = -np.sum(hist_range * np.log(hist_range))  # background entropy

        # Foreground/Object (bright)
        hist_range = data[it + 1:]
        # normalize within selected range & remove all 0 elements
        hist_range = hist_range[hist_range != 0] / (cdf[last_bin] - cdf[it])
        tot_ent -= np.sum(hist_range * np.log(hist_range))  # accumulate object entropy

        # find max
        if tot_ent > max_ent:
            max_ent, threshold = tot_ent, it

    return(threshold)

def max_entropy_threshold(I):
    """
    Function based on the previous one that directly takes an image for argument.
    """
    H, bins = exposure.histogram(I, nbins=256, source_range='image', normalize=False)
    T = max_entropy(H)
    return(T)

# %%% Physics

def computeMag_M270(B):
    M = 0.74257*1.05*1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
    return(M)

def computeMag_M450(B):
    M = 1.05*1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
    return(M)

def chadwickModel(h, E, H0, DIAMETER):
    R = DIAMETER/2
    f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    return(f)

def inversedChadwickModel(f, E, H0, DIAMETER):
    R = DIAMETER/2
    h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    return(h)

def getDimitriadisCoefs(v, order):
    a0 = -0.347*(3 - 2*v)/(1 - v)
    b0 = 0.056*(5 - 2*v)/(1 - v)
    
    ks = []
    if order >= 0:
        k0 = 1
        ks.append(k0)
        
        if order >= 1:
            k1 = - 2*a0/np.pi
            ks.append(k1)
        
            if order >= 2:
                k2 = + 4*a0**2/np.pi
                ks.append(k2)
            
                if order >= 3:
                    k3 = - (8/np.pi**3) * (a0**3 + (4*np.pi**2/15)*b0)
                    ks.append(k3)
            
                    if order >= 4:
                        k4 = + (16*a0/np.pi**4) * (a0**3 + (3*np.pi**2/5)*b0)
                        ks.append(k4)    
    return(ks)


def dimitriadisModel(h, E, H0):
    DIAMETER = 4477
    R = DIAMETER/2
    
    v = 0
    order = 2
    ks = getDimitriadisCoefs(v, order)
    
    delta = H0-h
    X = np.sqrt(R*delta)/h
    
    poly = np.zeros_like(X)
    
    for i in range(order+1):
        poly = poly + ks[i] * X**i
        
    F = ((4 * E * R**0.5 * delta**1.5)/(3 * (1 - v**2))) * poly
    
    return(F)
 
                    
# %%% Very general functions

def findFirst(x, A):
    """
    Find first occurence of x in array A, in a VERY FAST way.
    If you like weird one liners, you will like this function.
    """
    idx = (A==x).view(bool).argmax()
    return(idx)

def fitLine(X, Y):
    """
    returns: results.params, results \n
    Y=a*X+b ; params[0] = b,  params[1] = a
    
    NB:
        R2 = results.rsquared \n
        ci = results.conf_int(alpha=0.05) \n
        CovM = results.cov_params() \n
        p = results.pvalues \n
    
    This is how one should compute conf_int:
        bse = results.bse \n
        dist = stats.t \n
        alpha = 0.05 \n
        q = dist.ppf(1 - alpha / 2, results.df_resid) \n
        params = results.params \n
        lower = params - q * bse \n
        upper = params + q * bse \n
    """
    
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    params = results.params 
#     print(dir(results))
    return(results.params, results)


def toList(x):
    """
    if x is a list, return x
    if x is not a list, return [x]
    
    Reference
    ---------
    https://docs.python.org/3/library/collections.abc.html
    """
    t1 = isinstance(x, str) # Test if x is a string
    if t1: # x = 'my_string'
        return([x]) # return : ['my_string']
    else:
        t2 = isinstance(x, Collection) # Test if x is a Collection
        if t2: # x = [1,2,3] or x = array([1, 2, 3]) or x = {'k1' : v1}
            return(x) # return : x itself
        else: # x is not a Collection : probably a number or a boolean
            return([x]) # return : [x]
        
def toListOfStrings(x):
    """
    if x is a list, return x with all elements converted to string
    if x is not a list, return ['x']
    
    Reference
    ---------
    https://docs.python.org/3/library/collections.abc.html
    """
    t1 = isinstance(x, str) # Test if x is a string
    if t1: # x = 'my_string'
        return([x]) # return : ['my_string']
    else:
        t2 = isinstance(x, Collection) # Test if x is a Collection
        if t2: # x = [1,2,3] or x = array([1, 2, 3]) or x = {'k1' : v1}
            xx = [str(xi) for xi in x]
            return(xx) # return : x itself
        else: # x is not a Collection : probably a number or a boolean
            return([str(x)]) # return : [x]
          
def drop_duplicates_in_array(A):
    val, idx = np.unique(A, return_index = True)
    idx.sort()
    A_filtered = np.array([A[idx[j]] for j in range(len(idx))])
    return(A_filtered)

# %%% Figure & graphic operations

def simpleSaveFig(fig, name, savePath, ext, dpi):
    figPath = os.path.join(savePath, name + ext)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    fig.savefig(figPath, dpi=dpi)
    

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return(colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]))

# %%% User Input

class ChoicesBox(Qtw.QMainWindow):
    def __init__(self, choicesDict, title = 'Multiple choice box'):
        super().__init__()
        
        
        self.choicesDict = choicesDict
        self.questions = [k for k in choicesDict.keys()]
        self.nQ = len(self.questions)
        
        self.res = {}
        self.list_rbg = [] # rbg = radio button group

        self.setWindowTitle(title)
        
        layout = Qtw.QVBoxLayout()  # layout for the central widget
        main_widget = Qtw.QWidget(self)  # central widget
        main_widget.setLayout(layout)
        
        for q in self.questions:
            choices = self.choicesDict[q]
            label = Qtw.QLabel(q)
            layout.addWidget(label)
            rbg = Qtw.QButtonGroup(main_widget)
            for c in choices:
                rb = Qtw.QRadioButton(c)
                rbg.addButton(rb)
                layout.addWidget(rb)
                
            self.list_rbg.append(rbg)
            layout.addSpacing(20)
        
        valid_button = Qtw.QPushButton('OK', main_widget)
        layout.addWidget(valid_button)
        
        self.setCentralWidget(main_widget)
        
        valid_button.clicked.connect(self.validate_button)


    def validate_button(self):
        array_err = np.array([rbg.checkedButton() == None for rbg in self.list_rbg])
        Err = np.any(array_err)
        if Err:
            self.error_dialog()
        else:
            for i in range(self.nQ):
                q = self.questions[i]
                rbg = self.list_rbg[i]
                self.res[q] = rbg.checkedButton().text()
                
            self.quit_button()
            
    def error_dialog(self):
        dlg = Qtw.QMessageBox(self)
        dlg.setWindowTitle("Error")
        dlg.setText("Please make a choice in each category.")
        dlg.exec()
        
    def quit_button(self):
        Qtw.QApplication.quit()
        self.close()


def makeChoicesBox(choicesDict):
    """
    Create and show a dialog box with multiple choices.
    

    Parameters
    ----------
    choicesDict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    app = Qtw.QApplication(sys.argv)
    
    box = ChoicesBox(choicesDict)
    box.show()
        
    app.exec()
    res = box.res
    return(res)




