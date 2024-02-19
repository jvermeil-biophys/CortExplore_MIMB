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
from scipy import interpolate


# %% (1) Utility functions

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


def resize_2Dinterp(I, new_nx=None, new_ny=None, fx=None, fy=None):
    
    nX, nY = I.shape[1], I.shape[0]
    X, Y = np.arange(0, nX, 1), np.arange(0, nY, 1)
    try:
        newX, newY = np.arange(0, nX, nX/new_nx), np.arange(0, nY, nY/new_ny)
    except:
        newX, newY = np.arange(0, nX, 1/fx), np.arange(0, nY, 1/fy)
        
    # print(X.shape, Y.shape, newX.shape, newY.shape, I.shape)
    # fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
    # depthoHD = fd(XX, ZZ_HD)
    
    fd = interpolate.RectBivariateSpline(Y, X, I)
    newYY, newXX = np.meshgrid(newY, newX, indexing='ij')
    new_I = fd(newYY, newXX, grid=False)
    return(new_I)



