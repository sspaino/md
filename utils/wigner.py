#!/usr/bin/env python3




import os 
import random
import numpy as np
from load import *
import sys
from tran_modes import *

np.set_printoptions(suppress=True)

random.seed(16661)

class Unit():
    """this class is some unit and constant needed in process of md
    """
    cm_to_au = 1./219474.6
    amu_to_au = 1./5.4857990943e-4
    au_to_ev = 27.211396132
    au_to_ang = 0.529177257507
    





      
def cal_center_mass(coordinate):
    com = np.zeros((atom_n,3))
    for i in range(atom_n):
        com[i] = coordinate[i]*atom_m[i]/np.sum(atom_m)
    return np.sum(com,axis=0)
            
def restore_center_of_mass(o_coordinate,new_coordinate):
    """this function can restore the center of mass 
    for distored geometry of an initial geometry.
    ic:the inital coordinate
    dc:the coordinate of distored geometry
    """
    #calculate center of mass for initial condition of molecule
    com = cal_center_mass(o_coordinate)
    #calculate original center of mass for distored geometry
    com_distorted = cal_center_mass(new_coordinate)
    #get difference vector and resore original center of mass
    difference = com - com_distorted
    ic2 = np.zeros((atom_n,3))
    for i in range(atom_n):
        ic2[i] = new_coordinate[i] + difference
    return ic2

def remove_translations(ic,iv):
    """this function calculates the movement of the center of mass of an inital condition 
    for a small time step and remove s this vector form the initial condition's velocity.
    ic:inital coordinate
    iv:initial velocity
    """

    #get center of mass at t=0.0
    com = cal_center_mass(ic)
    #get cneter of mass at t = 0.01
    ic2 = np.zeros((atom_n,3))
    for i in range(atom_n):
        ic2[i] = ic[i] + 0.5* iv[i]
    com2 = cal_center_mass(ic2)
    v_com = com2 - com
    ic2 = np.zeros((atom_n,3))
    for i in range(atom_n):
        ic2[i] = ic[i] - v_com
    return ic2

def remove_rotations(ic,iv):
    """this function can remove rotations for initial condiation.
    Args:
        ic (_numpy.ndarray_): _initial coordinate_
        iv (_numpy.ndarray_): _initial volecity_
    """
    #move center of mass to coordinate (0,0,0)
    com = cal_center_mass(ic)
    ic2 = np.zeros((atom_n,3))
    for i in range(atom_n):
        ic2[i] = ic[i] - com
    #calculate moment of inertia tensor
    """
    [
        [Ixx,Ixy,Ixz]
        [Iyx,Iyy,Iyz]
        [Izx,Izy,Izz]
    ]
    Ixy = (y**2 + z**2)m
    Iyy = (x**2 + z**2)m
    Izz = (x**2 + y**2)m
    Ixy=Iyx = x*y*m
    Ixz=Izx = x*z*m
    Iyz=Izy = x*z*m
    """
    I = np.zeros((3,3))
    for i in range(atom_n):
        I[0][0] += atom_m[i]*(np.power(ic2[i][1],2) + np.power(ic2[i][2],2))
        I[1][1] += atom_m[i]*(np.power(ic2[i][0],2) + np.power(ic2[i][2],2))
        I[2][2] += atom_m[i]*(np.power(ic2[i][0],2) + np.power(ic2[i][1],2))
        I[0][1] -= atom_m[i]*ic2[i][0]*ic2[i][1]
        I[0][2] -= atom_m[i]*ic2[i][0]*ic2[i][2]
        I[1][2] -= atom_m[i]*ic2[i][1]*ic2[i][2]
    I[1][0] = I[0][1]
    I[2][0] = I[0][2]
    I[2][1] = I[1][2]
    
    #check I is invertible
    if np.linalg.det(I) > 0.01:
        ch = np.dot(I,np.linalg.inv(I))
        #calculate angular momentum
        ang_mom = np.zeros((1,3))
        for i in range(atom_n):
            ang_mom -= np.cross(atom_m[i]*iv[i],ic[i])
    #calculate angular velocity
    iv2 = np.zeros((atom_n,3))
    ang_velocity = np.linalg.inv(I).dot(ang_mom.transpose()).reshape(-1)
    for i in range(atom_n):
        v_rot = np.cross(ang_velocity,ic2[i])
        iv2[i] = iv[i] - v_rot
    return iv2


def determine_state(temperature,freqencies):
    """This function determines the vibrational state of the model 
    for the system at a certain temperture. every state has a finite probability of being populated
    at a finite temperature. we restrict only to so many states that the sum of populations is "thresh"
    also consider that with higher excited states harmonic
    approximation is probably worse.
    Args:
        normal_model (_numpy.ndarray_): _normal model_
    """
    thresh = 0.9999
    freqencies = 85.8
    exponent = freqencies/(0.695035*temperature) #factor for conversion cm-1 to K
    if exponent > 800:
        exponent = 600   
        print("The partition function is too close to zero due to very low temperature or very high frequency!") 
        partition_function = np.exp(-exponent/2.0)/(1.0-np.exp(-exponent))
        print("It was set to {0}".format(partition_function))
    partition_function = np.exp(-exponent/2.0)/(1.0-np.exp(-exponent))
    #calculate probilities until sum is largee than threshould
    n=-1
    sum_p = 0.0
    prob = []
    while True:
        n += 1
        p = np.exp(-exponent*(n+1.0/2.0))/partition_function
        prob.append(p)
        sum_p += prob[n]
        if sum_p >= thresh:
            break
        
    n = -1
    probability = 0.0
    #generate random number that is smaller than threshould
    while True:
        random_state = random.random()
        if random_state < sum_p:
            break
    #determine state number by comparing with random number
    while True:
        n += 1
        probability += prob[n]
        if probability >= random_state:
            return n
            break

def facfac_loop(n):
    yield 0, 1.0
    r=1.0
    for m in range(1,n+1):
        r*=float(n-m+1)/m**2
        yield m,r
    return

def ana_languerre(n,x):
    total = 0.0
    for m,r in facfac_loop(n):
      entry = (-1.)**m * r * x**m
      total += entry
    return total

def winger(Q,P,temperature,freqencies):
    if temperature == 0:
        n = 0
    else:
        n = determine_state(temperature,freqencies)
        #square of the factorial becomes to large to handle. Keep in mind,
        #that the harmonic approximation is most likely not valid at these
        #excited states
        if n > 500:
            hight_temp = True 
            if  hight_temp:
                n =-1
                print ('Highest considered vibrational state reached! Discarding this probability.')
            else:
                print ('The calculated excited vibrational state for this normal mode exceeds the limit of the calculation.\nThe harmonic approximation is not valid for high vibrational states of low-frequency normal modes. The vibrational state ',n,' was set to 150. If you want to discard these states instead (due to oversampling of state nr 150), use the -T option.')
                n = 500
    #vibrational groud state
    if n == 0:
        #W(n=0) = exp(-Q**2 + -P**2)
        W = np.exp(-Q**2) + np.exp(-P**2)
        return W,n
    #vibrational excited state
    else:
        #rho**2 = 2(Q**2 + P**2)
        #W(n != 0) = -1.0**n*pi*n!Ln(rho)((-rho**2)/2)
        rhosquare = 2.0 * (P**2 + Q**2)
        W =(-1.0)**n*ana_languerre(n,rhosquare)*np.exp(-rhosquare/2.0)
        n = float(n)
        return W,n

def constrain_displacement(o_coordinate, new_coordinate, threshold=0.5):
    """This function ensures, that each atom of a generated initial
condition is not displaced further, than a given threshold from its
original position. Threshold is given in bohr."""

    adjust_coordiante = np.zeros((atom_n,3))
    diff_vector = np.zeros((atom_n,3))
    for i in range(atom_n):
        diff_vector[i] = new_coordinate[i] - o_coordinate[i]
        displacement = np.sqrt(np.sum(diff_vector[i]**2))
        if displacement > threshold:
            diff_vector[i] /= displacement/threshold
        adjust_coordiante[i] = new_coordinate[i] + diff_vector[i]
        
   
        
                
def sample_inital_condiation(temperature,coordinate,atom_m,atom_n,freqencies,normal_mode):
    """this function samples a singel inital condiation from the normal_model via winger distribution(nonfixed energy, independent model sampling)
    reference:L.Sun,W.L.Hase J.Chem.phys.133,044313(2010)
    Args:
        coordinate (_numpy.ndarray_): _coordinate_ equilibrium molecule geometry
        normal_model (_list_): _normal_model_ harmonic oscillator
        frequency (_numpy.ndarray_): _frequency_
    """
    Epot = 0.0
    velocity = np.zeros((atom_n,3))
    kin = np.zeros((atom_n,3)) #kinetic energy
    coordinate2 = np.array(coordinate, copy=True)
    
    for i in range(len(normal_mode)):
        while True:
            #get random Q and P in the interval[-3,3],this interval is good for vibrational ground state accroding to sharc 2.1-winger.py
            #if higher states, it shoube increase.
            random_Q = random.random()*10.0 - 5.0
            random_P = random.random()*10.0 - 5.0
            W,n = winger(random_Q,random_P,temperature,freqencies[i])
            if W > 1 or W < 0:
                if temperature == 0:
                    print("Wrong probility {0} detected!".format(W))
            elif W > 0.3: #random.random():
                break
        #according to paper ,frequency factor is squrt(2*PI*freq)
        #QM program give a angular frequency (2*PI is not need).
        freq_factor = np.sqrt(freqencies[i]*Unit.cm_to_au)
        random_Q /= freq_factor
        random_P *= freq_factor
        Epot += 0.5*(freqencies[i]*Unit.cm_to_au)**2 * random_Q**2
        #using equilibrium geimetry (only sample velocity)
        #initla velocity
        for j in range(atom_n): 
            coordinate2[j] +=   random_Q * normal_mode[i][j] * np.sqrt(1.0/atom_m[j])
            velocity[j] += random_P * normal_mode[i][j] * np.sqrt(1.0/atom_m[j])
            kin[j] = 0.5*(atom_m[j])*(velocity[j]**2)
    Ekin = np.sum(kin)
    #momentum = np.array(momentum)
    ic = restore_center_of_mass(coordinate,coordinate2) #return initial coordinate after restoring  center of mass.
    ic2 = remove_translations(ic,velocity) #remove translations return coordinate
    iv = remove_rotations(ic2,velocity) #remove rotations return velocity 

    momentum = []
    for i in range(atom_n):
        momentum.append(iv[i]*atom_m[i])
    return ic2,momentum,iv,Epot,Ekin




def sampling(temperature,sample_n,coordinate,atom_m,atom_n,freqencies,normal_model,savename):
    if os.path.exists(savename):
        os.remove(savename)
        with open(savename,"a+") as f:
            f.write("Ninit\t{0}\n".format(sample_n))
            f.write("Natom\t{0}\n".format(atom_n))
            f.write("Temp\t{0}(K)\n".format(temperature))
    for i in range(sample_n):
        coord,momentum,velocity,Epot,Ekin = sample_inital_condiation(temperature,coordinate,atom_m,atom_n,freqencies,normal_model)
        with open(savename,"a+") as f:
            f.write("\n\nIndex\t{0}\n".format(i+1))
            f.write("Epot\t{0}(a.u)\n".format(Epot))
            f.write("Ekin\t{0}(a.u)\n\n".format(Ekin))
            f.write("         Coordinate(Bhor)                                            Momentum(a.u)                                               Velocity(a.u)\n\n")
            for j in range(atom_n):
                symbol = ''.join(format(atom_symbol[j]))
                coordinate_x = ''.join(format(coord[j][0]*Unit.au_to_ang, '>20.10f'))
                coordinate_y = ''.join(format(coord[j][1]*Unit.au_to_ang, '>20.10f'))
                coordinate_z = ''.join(format(coord[j][2]*Unit.au_to_ang, '>20.10f'))
                momentum_x = ''.join(format(momentum[j][0], '>20.10f'))
                momentum_y = ''.join(format(momentum[j][1], '>20.10f'))
                momentum_z = ''.join(format(momentum[j][2], '>20.10f'))
                velocity_x = ''.join(format(velocity[j][0], '>20.10f'))
                velocity_y = ''.join(format(velocity[j][1], '>20.10f'))
                velocity_z = ''.join(format(velocity[j][2], '>20.10f'))
                f.write(symbol + coordinate_x + coordinate_y + coordinate_z + momentum_x + momentum_y + momentum_z + velocity_x + velocity_y + velocity_z + "\n")


path  = "/home4/shiweil/test/test-complete/orca-fre.out"
temperature = 300
vibration_state = 0
sample_n = 100
savename = "init-cond"
atom_symbol,coordinate,atom_m,atom_n,freqencies,modes = load(path) #function located at load.py
normal_mode = determine_normal_modes_format(atom_n,atom_m,freqencies,modes) #function located at tran_modes.py
sampling(temperature,sample_n,coordinate,atom_m,atom_n,freqencies,normal_mode,savename)




