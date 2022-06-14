#!/usr/bin/env python3
import os 
from utils.surface_hopping import surface_hopping
sh = surface_hopping(
    work_dir = os.getcwd(),
    program_name = "ORCA",    #inital input
    init_input = "orca.inp" ,  #initial output
    init_output = "orca.out", #initiatl velocity
    init_momentum = "initial_condition", #initiatl velocity
    atom_n = 17,    #number of atoms in system
    total_time = 500, #total time of trajectory
    step_time = 0.5,
    state_n = 3, #number of states calculated in program
    hopping_threshold_value = 0.3, #Thresh energy for hops(ev)
)
sh.on_the_fly()