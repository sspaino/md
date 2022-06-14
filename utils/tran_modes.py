
import numpy as np
from load import *
low_freq = 10
amu_to_au = 1./5.4857990943e-4
ang_to_bohr = 1./0.529177211

def determine_normal_modes_format(atom_n,atom_m,frequencies,normal_mode):
    """
    modes:nornal model
    molecule:coordinate
    nmodes:the number of normal model
    
    This function determines the input format of the normal modes by trying to
    transform them to mass-weighted coordinates and seeing which of the four methods
    was able to do so via checking if the normal modes are now orthogonal. The mass-
    weighted normal coordinates are then returned
    """
    print ('\nStarting normal mode format determination...')
    #generate different set of modes that each undergo a different transformation
    #modes_1, modes_2, modes_3 and modes are represented by the numbers 1, 2, 3
    #and 4, where 1 stands for gaussian-type coordinates, 2 for cartesian coordinates,
    #3 for Colombus-type coordinates and 4 for already mass-weighted coordinates.
    #normformat = ["gaussian-type (Gaussian, Turbomole, Q-Chem, ADF, Orca)","cartesian (Molpro, Molcas)","columbus-type (Columbus)","mass-weighted"]
    normal_mode_1 = [] #gaussian-type (Gaussian, Turbomole, Q-Chem, ADF, Orca)
    normal_mode_2 = [] #"cartesian (Molpro, Molcas)"
    normal_mode_3 = [] #"columbus-type (Columbus)"
    normal_mode_4 = [] # mass-weighted
    modes_all = []
    #modes_4  I don't transform to mass-weighted herep
    #apply transformations to normal modes
    
    for i in range(len(normal_mode)):
        norm = np.zeros((atom_n,3))
        for j in range(atom_n):
            norm[j] = normal_mode[i][j]**2*atom_m[j]/amu_to_au
        norm = np.sqrt(np.sum(norm))
        if norm == 0.0 and frequencies[i] >= low_freq:
            print("warning:")
            print('WARNING: Displacement vector of mode {0} is null vector. Ignoring this mode!'.format(i+1))
            frequencies[i] == 0.0
        modes_1 = np.zeros((atom_n,3))
        modes_2 = np.zeros((atom_n,3))
        modes_3 = np.zeros((atom_n,3))
        modes_4 = np.zeros((atom_n,3))
        for j in range(atom_n):
            modes_1[j] = normal_mode[i][j]/(norm/np.sqrt(atom_m[j]/amu_to_au)) #gaussian-type (Gaussian, Turbomole, Q-Chem, ADF, Orca)
            modes_2[j] = normal_mode[i][j]*np.sqrt(atom_m[j]/amu_to_au) #"cartesian (Molpro, Molcas)"
            modes_3[j] = normal_mode[i][j]*np.sqrt(atom_m[j]/amu_to_au)/np.sqrt(ang_to_bohr) #"columbus-type (Columbus)" 
            modes_4[j] = normal_mode[i][j]
        #modes_4  I don't transform to mass-weighted here
        normal_mode_1.append(modes_1)
        normal_mode_2.append(modes_2)
        normal_mode_3.append(modes_3)
        normal_mode_4.append(modes_4)
    normal_mode_1 = np.array(normal_mode_1)
    normal_mode_2 = np.array(normal_mode_2)
    normal_mode_3 = np.array(normal_mode_3)
    normal_mode_4 = np.array(normal_mode_4)
    modes_all.append(normal_mode_1)
    modes_all.append(normal_mode_2)
    modes_all.append(normal_mode_3)
    modes_all.append(normal_mode_4)
    #create dotproduct matrices of the normal mode multiplication
    #for all four transformations.
    
    matrix_1 = []
    matrix_2 = []
    matrix_3 = []
    matrix_4 = []
    for i in range(len(normal_mode)):
        matrix_1.append(normal_mode_1[i].reshape(-1))
        matrix_2.append(normal_mode_2[i].reshape(-1))           
        matrix_3.append(normal_mode_3[i].reshape(-1)) 
        matrix_4.append(normal_mode_4[i].reshape(-1))
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    matrix_3 = np.array(matrix_3)
    matrix_4 = np.array(matrix_4)

    
    results = []
    results.append(np.dot(matrix_1,matrix_1.T))
    results.append(np.dot(matrix_2,matrix_2.T))
    results.append(np.dot(matrix_3,matrix_3.T))
    results.append(np.dot(matrix_4,matrix_4.T))
    #check for orthogonal matrices
    diagonalcheck = [[],[]]
    thresh = 0.05
    for result in results:
        trace = 0
        for i in range(len(result)):
            trace += result[i][i]
            result[i][i] -= 1
        diagonalcheck[0].append(trace)
        if any( [abs(i) > thresh for j in result for i in j ] ):
            diagonalcheck[1].append(0)
        else:
            diagonalcheck[1].append(1)
    possibleflags = []
    for i in range(4):
        if diagonalcheck[0][i] > len(normal_mode)-1 and diagonalcheck[0][i]/len(normal_mode)-1 < thresh and diagonalcheck[1][i] == 1:
            possibleflags.append(i+1)
            nm_flag = i

    normformat = ["gaussian-type (Gaussian, Turbomole, Q-Chem, ADF, Orca)","cartesian (Molpro, Molcas)","columbus-type (Columbus)","mass-weighted"]
    #check for input flag
    try:
      print("Final format specifier: {0} {1}".format(nm_flag+1, normformat[nm_flag]))
    except UnboundLocalError:
      print ("The normal mode analysis was unable to diagonalize the normal modes.")
      print ("Input is therefore neither in cartesian, gaussian-type, Columbus-type, or mass weighted coordinates.")
      exit(1)    
    if  len(possibleflags) == 1:
        print ("The normal modes input format was determined to be {0} coordinates.".format(normformat[nm_flag]))
        return modes_all[nm_flag]
    else:
        print("normal modes error,exit!")
        exit(1)
