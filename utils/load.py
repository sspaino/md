#!/usr/bin/env python3

import os 
import re
import numpy as np
 
"""
read frequencies  and normal model from orca.out 
"""


class Unit():
    """this class is some unit and constant needed in process of md
    """
    cm_to_au = 1./219474.6
    amu_to_au = 1./5.4857990943e-4
    au_to_ev = 27.211396132
    au_to_ang = 0.529177257507
    # atom mass
    periodic_table = {'H' :   1.007825 ,'He':   4.002603 ,'Li':   7.016004 ,'Be':   9.012182 ,'B' :  11.009305 ,'C' :  12.000000 ,'N' :  14.003074 ,'O' :  15.994915 ,
                      'F' :  18.998403 ,'Ne':  19.992440 ,'Na':  22.989770 ,'Mg':  23.985042 ,'Al':  26.981538 ,'Si':  27.976927 ,'P' :  30.973762 ,'S' :  31.972071 ,
                      'Cl':  34.968853 ,'Ar':  39.962383 ,'K' :  38.963707 ,'Ca':  39.962591 ,'Sc':  44.955910 ,'Ti':  47.947947 ,'V' :  50.943964 ,'Cr':  51.940512 ,
                      'Mn':  54.938050 ,'Fe':  55.934942 ,'Co':  58.933200 ,'Ni':  57.935348 ,'Cu':  62.929601 ,'Zn':  63.929147 ,'Ga':  68.925581 ,'Ge':  73.921178 ,
                      'As':  74.921596 ,'Se':  79.916522 ,'Br':  78.918338 ,'Kr':  83.911507 ,'Rb':  84.911789 ,'Sr':  87.905614 ,'Y' :  88.905848 ,'Zr':  89.904704 ,
                      'Nb':  92.906378 ,'Mo':  97.905408 ,'Tc':  98.907216 ,'Ru': 101.904350 ,'Rh': 102.905504 ,'Pd': 105.903483 ,'Ag': 106.905093 ,'Cd': 113.903358 ,
                      'In': 114.903878 ,'Sn': 119.902197 ,'Sb': 120.903818 ,'Te': 129.906223 ,'I' : 126.904468 ,'Xe': 131.904154 ,'Cs': 132.905447 ,'Ba': 137.905241 ,
                      'La': 138.906348 ,'Ce': 139.905435 ,'Pr': 140.907648 ,'Nd': 141.907719 ,'Pm': 144.912744 ,'Sm': 151.919729 ,'Eu': 152.921227 ,'Gd': 157.924101 ,
                      'Tb': 158.925343 ,'Dy': 163.929171 ,'Ho': 164.930319 ,'Er': 165.930290 ,'Tm': 168.934211 ,'Yb': 173.938858 ,'Lu': 174.940768 ,'Hf': 179.946549 ,
                      'Ta': 180.947996 ,'W' : 183.950933 ,'Re': 186.955751 ,'Os': 191.961479 ,'Ir': 192.962924 ,'Pt': 194.964774 ,'Au': 196.966552 ,'Hg': 201.970626 ,
                      'Tl': 204.974412 ,'Pb': 207.976636 ,'Bi': 208.980383 ,'Po': 208.982416 ,'At': 209.987131 ,'Rn': 222.017570 ,'Fr': 223.019731 ,'Ra': 226.025403 ,
                      'Ac': 227.027747 ,'Th': 232.038050 ,'Pa': 231.035879 ,'U' : 238.050783 ,'Np': 237.048167 ,'Pu': 244.064198 ,'Am': 243.061373 ,'Cm': 247.070347 ,
                      'Bk': 247.070299 ,'Cf': 251.079580 ,'Es': 252.082972 ,'Fm': 257.095099 ,'Md': 258.098425 ,'No': 259.101024 ,'Lr': 262.109692 ,
                      'Rf': 267. ,'Db': 268. ,'Sg': 269. ,'Bh': 270. ,'Hs': 270. ,'Mt': 278. ,'Ds': 281. ,'Rg': 282. ,'Cn': 285. ,'Nh': 286. ,'Fl': 289. ,'Mc': 290. ,
                      'Lv': 293. ,'Ts': 294. ,'Og': 294. ,
          } # atom mass




#get coordinate and atom_symbol
def get_coor_symbol(filename):
    with open(filename) as f:
        atom_symbol = []
        coordinate = [] #unit is a.u from orca.out
        for i in f:
            result = re.search("\s+\d+\s+[A-Z]{1,2}\s+\d+\.\d+\s+\d+\s+\d+\.\d+(\s+-?\d+\.\d+){3}",i)
            if result !=None:
                atom_symbol.append(result.group().split()[1])
                coordinate.append(result.group().split()[5:9])
        coordinate = np.array(coordinate).astype(float)
        atom_m = np.array([Unit.periodic_table[i]*Unit.amu_to_au  for i in atom_symbol]) .astype(float)
        atom_n = len(atom_m)
    return atom_symbol,coordinate,atom_m,atom_n


#get frequencies form orca.out
def get_freq(filename):
    with open(filename) as f:
        freq = []
        for i in f:
            result = re.search("\d+:\s+\d+\.\d+\s+cm\*\*-1",i)
            if result != None:
                freq.append(float(result.group().split()[1]))
    return freq
        
        
#get normal model form orca.out
def get_normal_model(filename,atom_n):
    with open(filename) as f:
        data1 = []
        data2 = []
        normal_model = []
        for i in f:
            result = re.search("\s{5,}\d+(\s+-?\d+.\d{6}){1,6}",i)
            if result !=None:
                if len(result.group().split()) == 7:
                    data1.append(result.group().split()[1:])
                else:
                    data2.append(result.group().split()[1:])
        data1 = np.array(data1).astype(float)
        data2 = np.array(data2).astype(float)
        m,n = data1.shape
        for i in range(int(m/(atom_n*3))):
            data3 = data1[i*(atom_n*3):(i+1)*(atom_n*3)]
            for j in range(n):
                normal_model.append(data3[:,j])
        m,n = data2.shape
        for i in range(int(m/(atom_n*3))):
            data4 = data2[i*(atom_n*3):(i+1)*(atom_n*3)]
            for  j in range(n):
                normal_model.append(data4[:,j])
    return normal_model


def molden_out(atom_symbol,coordinate,atom_n,freqencies,normal_model):
    #delet zero frequencies and related nornal modes
    non_zero_freq = []
    non_zero_modes = []
    for i in range(len(freqencies)):
        if freqencies[i] != 0:
            non_zero_freq.append(freqencies[i])
            non_zero_modes.append(normal_model[i])
    with open("test.molden","a+") as f:
        f.write(" [FREQ]\n")
        for i in range(len(non_zero_freq)):
                freq = ''.join(format(non_zero_freq[i], '>10.2f')) #unit cm-1
                f.write(freq +"\n")
        f.write(" [FR-COORD]\n")
        for i in range(atom_n):
                symbol = ''.join(format(atom_symbol[i]))
                coordinate_x = ''.join(format(coordinate[i][0], '>20.10f')) #unit a.u
                coordinate_y = ''.join(format(coordinate[i][1], '>20.10f'))
                coordinate_z = ''.join(format(coordinate[i][2], '>20.10f'))
                f.write(symbol + coordinate_x + coordinate_y + coordinate_z +"\n")
        f.write(" [FR-NORM-COORD]\n")
        for i in range(len(non_zero_modes)):
            f.write(" Vibration                     {0}\n".format(i+1))
            for j in range(atom_n):
                coordinate_x = ''.join(format(non_zero_modes[i].reshape(atom_n,3)[j][0], '>20.10f'))
                coordinate_y = ''.join(format(non_zero_modes[i].reshape(atom_n,3)[j][1], '>20.10f'))
                coordinate_z = ''.join(format(non_zero_modes[i].reshape(atom_n,3)[j][2], '>20.10f'))
                f.write(coordinate_x + coordinate_y + coordinate_z +"\n")

def load(filename):
    atom_symbol,coordinate,atom_m,atom_n = get_coor_symbol(filename)
    freqencies = get_freq(filename)
    normal_model = get_normal_model(filename,atom_n)
    #delet zero frequencies and related nornal modes
    non_zero_freq = []
    non_zero_modes = []
    for i in range(len(freqencies)):
        if freqencies[i] != 0:
            non_zero_freq.append(freqencies[i])
            non_zero_modes.append(np.array(normal_model[i]).reshape(atom_n,3))
    non_zero_freq = np.array(non_zero_freq)
    non_zero_freq = np.array(non_zero_freq)
    return atom_symbol,coordinate,atom_m,atom_n,non_zero_freq,non_zero_modes







                    
