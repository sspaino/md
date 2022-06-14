#!/usr/bin/env python3
import os
import re
import shutil
import numpy as np
np.set_printoptions(suppress=True)






class Unit():
    """this class is some unit and constant needed in process of md
    """
    au_to_fs = 0.0241888439241
    amu_to_au = 1822.88853006
    au_to_ev = 27.2113956555
    au_to_ang = 0.529177257507
    pi = 3.14159265358979323846
    h = 6.626075499999999e-34
    h_bar =  1
    # atom mass
    periodic_table = {'X': 0, 'Ac': 227.028, 'Al': 26.981539, 'Am': 243, 'Sb': 121.757, 'Ar': 39.948, 'As': 74.92159, 'At': 210,
          'Ba': 137.327, 'Bk': 247, 'Be': 9.012182, 'Bi': 208.98037, 'Bh': 262, 'B': 10.811, 'Br': 79.904,
          'Cd': 112.411, 'Ca': 40.078, 'Cf': 251, 'C': 12.011, 'Ce': 140.115, 'Cs': 132.90543, 'Cl': 35.4527,
          'Cr': 51.9961, 'Co': 58.9332, 'Cu': 63.546, 'Cm': 247, 'Db': 262, 'Dy': 162.5, 'Es': 252, 'Er': 167.26,
          'Eu': 151.965, 'Fm': 257, 'F': 18.9984032, 'Fr': 223, 'Gd': 157.25, 'Ga': 69.723, 'Ge': 72.61,
          'Au': 196.96654, 'Hf': 178.49, 'Hs': 265, 'He': 4.002602, 'Ho': 164.93032, 'H': 1.00794, 'In': 114.82,
          'I': 126.90447, 'Ir': 192.22, 'Fe': 55.847, 'Kr': 83.8, 'La': 138.9055, 'Lr': 262, 'Pb': 207.2, 'Li': 6.941,
          'Lu': 174.967, 'Mg': 24.305, 'Mn': 54.93805,
          'Mt': 266, 'Md': 258, 'Hg': 200.59, 'Mo': 95.94, 'Nd': 144.24, 'Ne': 20.1797, 'Np': 237.048, 'Ni': 58.6934,
          'Nb': 92.90638, 'N': 14.00674, 'No': 259, 'Os': 190.2, 'O': 15.9994, 'Pd': 106.42, 'P': 30.973762,
          'Pt': 195.08, 'Pu': 244, 'Po': 209, 'K': 39.0983, 'Pr': 140.90765, 'Pm': 145, 'Pa': 231.0359, 'Ra': 226.025,
          'Rn': 222, 'Re': 186.207, 'Rh': 102.9055, 'Rb': 85.4678, 'Ru': 101.07, 'Rf': 261, 'Sm': 150.36,
          'Sc': 44.95591, 'Sg': 263,
          'Se': 78.96, 'Si': 28.0855, 'Ag': 107.8682, 'Na': 22.989768, 'Sr': 87.62, 'S': 32.066, 'Ta': 180.9479,
          'Tc': 98, 'Te': 127.6, 'Tb': 158.92534, 'Tl': 204.3833, 'Th': 232.0381, 'Tm': 168.93421, 'Sn': 118.71,
          'Ti': 47.88, 'W': 183.85, 'U': 238.0289, 'V': 50.9415, 'Xe': 131.29, 'Yb': 173.04, 'Y': 88.90585, 'Zn': 65.39,
          'Zr': 91.224} # atom mass

class MOLPRO():
    def __init__(self,atom_n,init_input,init_momentum,state_n):
        self.atom_n = atom_n
        self.init_input = init_input
        self.init_momentum = init_momentum
        self.state_n = state_n
    
    def atom_symbol(self):
        """
        get atom_symbol
        """
        with open(self.init_momentum) as f:
            atom_symbol = []
            for i in f:
                result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                if result != None:
                    atom_symbol.append(result.group().split()[0])
            return atom_symbol
    
    def atom_m(self):
        """
        aotm mass for eacho element of system(a.u.)
        """
        atom_mass_list = np.zeros((self.atom_n,1))
        for i in range(self.atom_n):
            atom_mass_list[i] = Unit.periodic_table[self.atom_symbol()[i]]*Unit.amu_to_au 
        return atom_mass_list

    def check(self,filename):
        """
        check file
        """
        if os.path.exists(filename):
            print("%s is exists"%filename)
        else:
            print ("%s is not exists,please check it!")

    def check_out(self,filename):
        """
        This is function for checing  output of molpro.
        filname:the name of output of molpro
        """
        with open(filename) as f:
            for i in f:
                result = re.search("CONVERGENCE REACHED, FINAL GRADIENT:",i)
                if result != None:
                    return result.group()
    
    def save_wavefunction_file(self,filename):
        """
        save wavefunction after run molpro
        """
        init_wfu_filename = self.init_input.split(".")[0] + ".wfu"
        wfu_filename = filename.split(".")[0] + ".wfu"
        command = "cp {0} {1}".format(init_wfu_filename,wfu_filename)
        os.system(command)
        print("save wavefunction file form {0} to {1}".format(init_wfu_filename,wfu_filename))
                       
    def run(self,filename):
        """
        command for runing molpro
        """
        print("\nStart ab initial calculation!")
        command ="molpro"+" " + "-W" + " " +"./"+ " " + filename +" --no-xml-output"
        os.system(command)
        if filename == self.init_input:
            print("Inital wavefunction file do not need to save!")
        else:
            self.save_wavefunction_file(filename)

    def printf(self,matrix,var_name,step,time):
        string =" {0} at step {1} Time = {2}(fs) ".format(var_name,str(step),str(time))
        print('\n{0:*^63}\n'.format(string))
        for i in range(self.atom_n):
            print("%2s%20.10f%20.10f%20.10f"%(self.atom_symbol()[i] ,matrix[i][0],matrix[i][1],matrix[i][2]))
            
    def coordinate(self,filename):
        """
        get coordinate matrix (a.u)
        """

        with open(filename) as f:
            coordinate_matrix = []
            for i in f:
                result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                if result != None:
                    coordinate_matrix.append([format(float(i), '>.10f') for i in result.group().split()[1:4]])
            return (np.array(coordinate_matrix).astype(float))/Unit.au_to_ang

    def momentum(self,filename):
        """
        get initial momentum matrix
        """
        with open(filename) as f:
            momentum_matrix = []
            for i in f:
                result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                if result != None:
                    momentum_matrix.append([format(float(i), '>.10f') for i in result.group().split()[1:4]])
            return np.array(momentum_matrix).astype(float)

    def gradient(self,filename):
        """
        get gradient matrix
        \d+\.\d
        """
        command = "grep -A"+str(3+self.atom_n) +" \"SA-MC GRADIENT FOR STATE\" " + filename
        data = os.popen(command).readlines()
        gradient_matrix = []
        for i in data:
            result = re.search('(\s+-?\d+\.\d+){3}',i)
            if result != None: 
                gradient_matrix.append([format(float(i), '>.10f') for i in  result.group().split()[0:3]])
        return np.array(gradient_matrix).astype(float)
    
    def spin(self,filename):
        """
        get spin
        """
        with open(filename) as f:
            for i in f:
                result = re.search("Spin(\s)symmetry=(\w+)",i)
                if result != None:
                    spin = result.group().split()[1].split("=")[1]
        return spin
    
    def state(self,filename):
        
        """
        get state
        """
        with open(filename) as f:
            for i in f:
                result = re.search("Solving(\s)MCSCF(\s)z-vector(\s)equations(\s)for(\s)state(\s){1,2}(\d\.\d)",i)
                if result != None:
                    state = result.group().split()[6]
        return state

    def energy(self,filename,state):
        """
        get energy
        all: if 
        """
        with open(filename) as f:
            data_1 = []
            data_2 = []
            for i in f:
                result = re.search("!MCSCF(\s)STATE(\s\d\.\d\s)Energy(\s+(-?\d+\.\d+))",i)
                if result != None:
                    state_symbol = result.group().split()[2]
                    state_energy =  result.group().split()[4]
                    data_1.append(state_symbol )
                    data_2.append(float(state_energy))
            energy_dict = dict(zip(data_1[0:self.state_n],data_2[0:self.state_n]))
            if state:
                return energy_dict[str(state)]
            elif state == False:
                return energy_dict

    def replace_coordinate(self,filename_1,filename_2,coordiante):
        """
        replace coordinate
        """
        coordiante = coordiante*Unit.au_to_ang
        atom_symbol = self.atom_symbol()
        f1 = open(filename_1)
        f2 = open(filename_2,"w+")
        n = 0
        for i in f1:
            result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
            if result == None:
                f2.write(i)
            if result != None:  
                symbol = ''.join(format(atom_symbol[n]))
                coordinate_1 = ''.join(format(coordiante[n][0], '>18.10f'))
                coordinate_2 = ''.join(format(coordiante[n][1], '>18.10f'))
                coordinate_3 = ''.join(format(coordiante[n][2], '>18.10f'))
                reps = re.sub(result.group(),symbol + coordinate_1 + coordinate_2 + coordinate_3,i)
                f2.write(reps)
                n += 1
        f1.close()
        f2.close()
        
    def replace_state(self,filename_1,filename_2,state):
        """
        replace state
        """
        f1 = open(filename_1,"r")
        f2 = open(filename_2,"w+")
        for i in f1:
            result1 = re.search("^cpmcscf,grad,\d\.\d,record=510\d\.\d;",i)
            result2 = re.search("^force;samc,510\d\.\d;",i)
            if result1 == None and result2 == None:
                f2.write(i)
            if result1 != None:
                reps_1 = re.sub(result1.group(),"cpmcscf,grad,{0},record=510{1};".format(state,state),i)
                f2.write(reps_1)
            if result2 != None:
                reps_2 = re.sub(result2.group(),"force;samc,510{0};".format(state),i)
                f2.write(reps_2)
        f1.close()
        f2.close()
    
    def replace_wfu_filename(self,filename_1,filename_2):
            
        f1 = open(filename_1,"r")
        f2 = open(filename_2,"w+")
        for i in f1:
            #file,2,molpro.wfu
            result = re.search("^file,2,\w+\.wfu$",i)
            if result == None:
                f2.write(i)
            if result != None:
                filename_wfu = filename_1.split(".")[0]
                reps = re.sub(result.group(),"file,2,{0}.wfu".format(filename_wfu),i)
                f2.write(reps)  
        f1.close()
        f2.close()
    
    def gen_filename(self,prefix):
        """
        generate filename 
        prefix: prefix
        """
        input = prefix + ".in"
        output = prefix + ".out"
        return input,output
    
    def remove_file(self,prefix): 
            os .remove(prefix + ".in")
            os .remove(prefix + ".out")
            os .remove(prefix + ".wfu")
       
class GAUSSINA():
    
        def __init__(self,atom_n,init_input,init_momentum,state_n):
            self.atom_n = atom_n
            self.init_input = init_input
            self.init_momentum = init_momentum
            self.state_n = state_n

        def atom_symbol(self):
            """
            get atom_symbol
            """
            with open(self.init_momentum) as f:
                atom_symbol = []
                for i in f:
                    result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                    if result != None:
                        atom_symbol.append(list(result.group())[0])
                return atom_symbol
        
        def atom_m(self):
            """
            aotm mass for eacho element of system(a.u.)
            """
            atom_mass_list = np.zeros((self.atom_n,1))
            for i in range(self.atom_n):
                atom_mass_list[i] = Unit.periodic_table[self.atom_symbol()[i]]*Unit.amu_to_au 
            return atom_mass_list

        def check(self,filename):
            """
            check file
            """
            if os.path.exists(filename):
                print("%s is exists"%filename)
            else:
                print ("%s is not exists,please check it!")

        def check_out(self,filename):
            """
            This is function for checing  output of gaussian.
            filname:the name of output of gaussian
            """
            with open(filename) as f: 
                for i in f:
                    # Normal termination of Gaussian 09 at Wed May 25 01:25:45 2022.
                    result = re.search("Normal termination of Gaussian",i)
                    if result != None:
                        return result.group()
        
        def save_wavefunction_file(self,filename):
            """
            save wavefunction after run molpro
            """
            init_wfu_filename = self.init_input.split(".")[0] + ".chk"
            wfu_filename = filename.split(".")[0] + ".chk"
            command = "cp {0} {1}".format(init_wfu_filename,wfu_filename)
            os.system(command)
            print("save wavefunction file form {0} to {1}".format(init_wfu_filename,wfu_filename))
               
        def run(self,filename):
            """
            command for runing gaussian
            """
            print("\nStart ab initial calculation!")
            filename = filename.split(".")[0]
            command ="g09 " + (filename + ".gjf") + " " + (filename + ".out")
            os.system(command)
            if filename == self.init_input.split(".")[0]:
                print("Inital wavefunction file do not need to save!")
            else:
                self.save_wavefunction_file(filename)
            
        def printf(self,matrix,var_name,step,time):
            string =" {0}at step {1} Time = {2}(fs) ".format(var_name,str(step),str(time))
            print('\n{0:*^63}\n'.format(string))
            for i in range(self.atom_n):
                print("%2s%20.10f%20.10f%20.10f"%(self.atom_symbol()[i] ,matrix[i][0],matrix[i][1],matrix[i][2]))
                
        def coordinate(self,filename):
            """
            get coordinate matrix (a.u)
            """

            with open(filename) as f:
                coordinate_matrix = []
                for i in f:
                    result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                    if result != None:
                        coordinate_matrix.append([format(float(i), '>.10f') for i in list(result.group().split())[1:4]])
                return (np.array(coordinate_matrix).astype(float))/Unit.au_to_ang

        def momentum(self,filename):
            """
            get initial momentum matrix
            """
            with open(filename) as f:
                momentum_matrix = []
                for i in f:
                    result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                    if result != None:
                        momentum_matrix.append([format(float(i), '>.10f') for i in list(result.group().split())[1:4]])
                return np.array(momentum_matrix).astype(float)

        def gradient(self,filename):
            """
            get gradient matrix
            \d+\.\d
            """
            command = "grep -A"+str(2+self.atom_n) +" \"Center     Atomic                   Forces (Hartrees/Bohr)\" " + filename
            data = os.popen(command).readlines()
            gradient_matrix = []
            for i in data:
                result = re.search('(\s+-?\d+\.\d+){3}',i)
                if result != None:
                    gradient_matrix.append([format(float(i), '>.10f') for i in (result.group().split())[0:4]])
            return -np.array(gradient_matrix).astype(float)
            
        def spin(self,filename): 
            """
            get spin
            """
            with open(filename) as f:
                for i in f:
                    result = re.search("Multiplicity(\s)=(\s)\d+",i)
                    if result != None:
                        spin = str(result).split()[6].split("'")[0]
            return spin
            
        def state(self,filename): 
            """
            get state
            """
            with open(filename) as f:
                for i in f:
                    result = re.search("root=\d+",i)
                    if result != None:
                        state = int(result.group().split("=")[1])+1
            return str(state)

        def energy(self,filename,state):
            """
            get energy
            """
            with open(filename) as f:
                state_symbol = [str(i) for i in range(1,self.state_n+1)]
                data_1 = []
                data_2 = []
                for i in f:
                    #SCF Done:  E(RB3LYP) =  -359.298459692
                    #Excited State   1:      Singlet-?Sym    3.0115 eV  411.71 nm  f=0.0006  <S**2>=0.000
                    result1 = re.search("SCF\sDone:\s+E\([A-Za-z0-9]+\)\s=\s+-?\d+\.\d+",i)
                    result2 = re.search("Excited\sState\s+\d+:\s+\w+-\?\w+\s+\d+\.\d+\seV\s+-?\d+\.\d+\s+nm\s+f=-?\d+\.\d+\s+\<\w+\*\*\d+\>=\d+\.\d+",i)
                    if result1 !=None:
                        SCF_Done_energy = float(result1.group().split()[4])
                        data_1.append(SCF_Done_energy)
                    elif result2 != None:
                        excited_energy = float(result2.group().split()[4])              
                        data_2.append(excited_energy)
                data_3 = data_1 + [(i/Unit.au_to_ev+data_1[0]) for i in data_2]
                energy_dict = dict(zip(state_symbol,data_3[0:self.state_n]))
                if state:
                    return energy_dict[str(int(state)-1)]
                elif state == False:
                    return energy_dict
   
        def replace_coordinate(self,filename_1,filename_2,coordiante):
            """
            replace coordinate
            """
            coordiante = coordiante*Unit.au_to_ang
            atom_symbol = self.atom_symbol()
            f1 = open(filename_1)
            f2 = open(filename_2,"w+")
            n = 0
            for i in f1:
                result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                if result == None:
                    f2.write(i)
                if result != None:
                    symbol = ''.join(format(atom_symbol[n]))
                    coordinate_1 = ''.join(format(coordiante[n][0], '>18.10f'))
                    coordinate_2 = ''.join(format(coordiante[n][1], '>18.10f'))
                    coordinate_3 = ''.join(format(coordiante[n][2], '>18.10f'))
                    reps = re.sub(result.group(),symbol + coordinate_1 + coordinate_2 + coordinate_3,i)
                    f2.write(reps)
                    n += 1
            f1.close()
            f2.close()
            
        def replace_state(self,filename_1,filename_2,state):
            """
            replace state
            """
            f1 = open(filename_1,"r")
            f2 = open(filename_2,"w+")
            for i in f1:
                #p guess=read  B3LYP STO-3G tda=(nstates=6, root=2)  force nosymm
                result = re.search("\#\w\s+guess=read\s+[A-Za-z0-9]+\s+\w+-?[A-Za-z0-9]+\s+tda=\(nstates=\d+,\s+root=\d+\)\s+force\s+nosymm",i)
                if int(state) == 1:
                    if result == None:
                        f2.write(i)
                    elif result != None:
                        reps = re.sub("\s+tda=\(nstates=\d+,\s+root=\d+\)\s","",i)
                        f2.write(reps)
                elif int(state) != 1:
                    if result == None:
                        f2.write(i)
                    elif result != None:
                        reps = re.sub("root=\d+","root={}".format(state-1),i)
                        f2.write(reps)
            f1.close()
            f2.close()
     
        def gen_filename(self,prefix):
            """
            generate filename 
            prefix: prefix
            """
            input = prefix + ".gjf"
            output = prefix + ".out"
            return input,output
                
        def replace_wfu_filename(self,filename_1,filename_2):        
            f1 = open(filename_1,"r")
            f2 = open(filename_2,"w+")
            for i in f1:
                #file,2,molpro.wfu
                result = re.search("\%\w+=\w+\.chk",i)
                
                if result == None:
                    f2.write(i)
                if result != None:
                    filename_wfu = filename_1.split(".")[0]
                    reps = re.sub(result.group(),"%chk={0}.chk".format(filename_wfu),i)
                    f2.write(reps)      
            f1.close()
            f2.close()
    
        def remove_file(self,prefix): 
            os .remove(prefix + ".gjf")
            os .remove(prefix + ".out")
            os .remove(prefix + ".chk")   
     
class ORCA():
    
    def __init__(self,atom_n,init_input,init_momentum,state_n):
        self.atom_n = atom_n
        self.init_input = init_input
        self.init_momentum = init_momentum
        self.state_n = state_n
        
    def atom_symbol(self):
        """
        get atom_symbol
        """
        with open(self.init_momentum) as f:
            atom_symbol = []
            for i in f:
                result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                if result != None:
                    atom_symbol.append(result.group().split()[0])
            return atom_symbol
    
    def atom_m(self):
        """
        aotm mass for eacho element of system(a.u.)
        """
        atom_mass_list = np.zeros((self.atom_n,1))
        for i in range(self.atom_n):
            atom_mass_list[i] = Unit.periodic_table[self.atom_symbol()[i]]*Unit.amu_to_au 
        return atom_mass_list

    def check(self,filename):
        """
        check file
        """
        if os.path.exists(filename):
            print("%s is exists"%filename)
        else:
            print ("%s is not exists,please check it!")

    def check_out(self,filename):
        """
        This is function for checing  output of molpro.
        filname:the name of output of molpro
        """
        with open(filename) as f:
            for i in f:
                result = re.search("\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*",i)
                if result != None:
                    return result.group()
    
    def save_wavefunction_file(self,filename):
        """
        save wavefunction after run molpro
        """
        init_wfu_filename = self.init_input.split(".")[0] + ".gbw"
        wfu_filename = filename.split(".")[0] + ".gbw"
        command = "cp {0} {1}".format(init_wfu_filename,wfu_filename)
        os.system(command)
        print("save wavefunction file form {0} to {1}".format(init_wfu_filename,wfu_filename))
                       
    def run(self,filename):
        """
        command for runing molpro
        """
        
        print("\nStart ab initial calculation!")
        filename = filename.split(".")[0]
        command = "/share/apps/opt/orca/orca_5_0_2_linux_x86-64_openmpi411/orca "+ (filename+".inp") +" > " +(filename+".out")
        os.system(command)
        shutil.copy("orca_tmp.gbw","orca.gbw")
        if filename == self.init_input.split(".")[0]:
            print("Inital wavefunction file do not need to save!")
        else:
            self.save_wavefunction_file(filename)
        
    def printf(self,matrix,var_name,step,time):
        string =" {0} at step {1} Time = {2}(fs) ".format(var_name,str(step),str(time))
        print('\n{0:*^63}\n'.format(string))
        for i in range(self.atom_n):
            print("%2s%20.10f%20.10f%20.10f"%(self.atom_symbol()[i] ,matrix[i][0],matrix[i][1],matrix[i][2]))
            
    def coordinate(self,filename):
        """
        get coordinate matrix (a.u)
        """

        with open(filename) as f:
            coordinate_matrix = []
            for i in f:
                result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                if result != None:
                    coordinate_matrix.append([format(float(i), '>.10f') for i in result.group().split()[1:4]])
            return (np.array(coordinate_matrix).astype(float))/Unit.au_to_ang

    def momentum(self,filename):
        """
        get initial momentum matrix
        """
        with open(filename) as f:
            momentum_matrix = []
            for i in f:
                result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
                if result != None:
                    momentum_matrix.append([format(float(i), '>.10f') for i in result.group().split()[1:4]])
            return np.array(momentum_matrix).astype(float)

    def gradient(self,filename):
        """
        get gradient matrix
        \d+\.\d
        """
        command = "grep -A"+str(2+self.atom_n) +" \"CARTESIAN GRADIENT\" " + filename
        data = os.popen(command).readlines()
        gradient_matrix = []
        for i in data:
            result = re.search('(\s+-?\d+\.\d+){3}',i)
            if result != None: 
                gradient_matrix.append([format(float(i), '>.10f') for i in  result.group().split()[0:3]])
        return np.array(gradient_matrix).astype(float)
    
    def spin(self,filename):
        """
        get spin
        """
        with open(filename) as f:
            for i in f:
                result = re.search("\*\s+xyz\s\d\s\d",i)
                if result != None:
                    spin = result.group().split()[3]
        return spin
    
    def state(self,filename):
        
        """
        get state
        """
        flag = None
        with open(filename.split(".")[0]+".inp","r") as f3:
            for i in f3:
                result = re.search("%TDDFT",i)
                if result != None:
                    flag = result
        if  flag != None:
            with open(filename) as f:
                for i in f:
                    result1 = re.search("DE\(CIS\)\s=\s+-?\d+\.\d+\sEh\s\(Root\s+\d+\)",i)
                    if result1 != None:
                        state = int(result1.group().split()[5].split(")")[0])+1      
        elif flag == None:
                    state = 1
        return state


            
    def replace_coordinate(self,filename_1,filename_2,coordiante):
        """
        replace coordinate
        """
        coordiante = coordiante*Unit.au_to_ang
        atom_symbol = self.atom_symbol()
        f1 = open(filename_1)
        f2 = open(filename_2,"w+")
        n = 0
        for i in f1:
            result = re.search('([a-zA-Z]{1,2})(\s+-?\d+\.\d+){3}',i)
            if result == None:
                f2.write(i)
            if result != None:  
                symbol = ''.join(format(atom_symbol[n]))
                coordinate_1 = ''.join(format(coordiante[n][0], '>18.10f'))
                coordinate_2 = ''.join(format(coordiante[n][1], '>18.10f'))
                coordinate_3 = ''.join(format(coordiante[n][2], '>18.10f'))
                reps = re.sub(result.group(),symbol + coordinate_1 + coordinate_2 + coordinate_3,i)
                f2.write(reps)
                n += 1
        f1.close()
        f2.close()
        
    def replace_state(self,filename_1,filename_2,state):
        """
        replace state
        switch:False,True
        """
        flag = None
        with open(filename_1) as f:
            for i in f:
                result = re.search("\%TDDFT",i)
                if result !=None:
                    flag = result 
                    
        f1 = open(filename_1,"r")
        f2 = open(filename_2,"w+")
        for i in f1:
            result1 = re.search("^iroot\s+\d+",i)
            result2 = re.search("^nroots\s+\d+",i)
            result3 = re.search("\%TDDFT",i)
            result4 = re.search("tda = \w+",i)
            result5 = re.search("printlevel 3 end",i)
            result6 = re.search("%pal nprocs 4 end",i)
            if state != 1:
                if result1 == None and flag !=None:
                    f2.write(i)
                if result1 != None:
                    reps_1 = re.sub(result1.group(),"iroot {0}".format(int(state-1)),i)
                    f2.write(reps_1)
                        
                        
                elif result3 == None and flag == None:
                    if result6 == None:
                        f2.write(i)
                    if result6 != None:
                        content1 = "%pal nprocs 4 end\n"
                        content2 = "%TDDFT\n" 
                        content3 = "nroots 5\n"
                        content4 = "iroot  {0}\n".format(str(int(state)-1))
                        content5 = "tda = false\n"
                        content6 = "printlevel 3 end\n" 
                        f2.write(content1)
                        f2.write(content2)
                        f2.write(content3)
                        f2.write(content4)
                        f2.write(content5)
                        f2.write(content6)
           
            elif state ==1:
                if result1 == None and result2 ==None and result3 ==None and result4 ==None and result5 ==None:
                    f2.write(i)
        f1.close()
        f2.close()
        

    def replace_wfu_filename(self,filename_1,filename_2):
            
        f1 = open(filename_1,"r")
        f2 = open(filename_2,"w+")
        for i in f1:
            #%moinp "orca.gbw"
            result = re.search("\%moinp\s+\"\w+.gbw\"",i)
            if result == None:
                f2.write(i)
            if result != None:
                filename_wfu = filename_1.split(".")[0]
                reps = re.sub(result.group(),"%moinp \"{0}.gbw\"".format(filename_wfu),i)
                f2.write(reps)  
        f1.close()
        f2.close()
    
    def gen_filename(self,prefix):
        """
        generate filename 
        prefix: prefix
        """
        input = prefix + ".inp"
        output = prefix + ".out"
        return input,output

    def remove_file(self,prefix): 
            os .remove(prefix + ".inp")
            os .remove(prefix + ".out")
            os .remove(prefix + ".gbw")


    def energy(self,filename,state):
        """
        get energy
        """
        flag = None
        with open(filename.split(".")[0]+".inp","r") as f3:
            for i in f3:
                result11 = re.search("%TDDFT",i)
                if result11 != None:
                    flag = result11
                    
        if flag != None :
            with open(filename) as f:
                state_symbol = [str(i) for i in range(1,self.state_n+1)]
                data1 = []
                data2 = []
                for i in f:
                    result1 = re.search("E\(SCF\)\s+=\s+-?\d+\.\d+\s+Eh",i)
                    result2 = re.search("STATE\s+\d+:\s+E=\s+-?\d+\.\d+\s+au\s+-?\d+\.\d+\s+eV\s+-?\d+\.\d+\s+cm\*\*-?\d+\s+\<\w+\*\*\d+\>\s+=\s+-?\d+\.\d+",i)
                    if result1 !=None:
                        SCF_Done_energy = float(result1.group().split()[2])
                        data1.append(SCF_Done_energy)
                    elif result2 != None:
                        excited_energy = float(result2.group().split()[3])              
                        data2.append(excited_energy)
                data_3 = data1 + [(i+data1[0]) for i in data2]
                energy_dict = dict(zip(state_symbol,data_3[0:self.state_n]))
                if state:
                    return energy_dict[str(int(state))]
                elif state == False:
                    return energy_dict
                        
        elif flag == None :
                temp_input = filename.split(".")[0]+"-stateup.inp"
                temp_output = filename.split(".")[0]+"-stateup.out"
                f1 = open(filename.split(".")[0]+".inp","r")
                f2 = open(temp_input,"w+")
                for i in f1:
                    result3 = re.search("%moinp \"orca.gbw\"",i)
                    result4 = re.search("%base \"orca_tmp\"",i)
                    result5 = re.search("%pal nprocs \d+ end",i)
                    if result3 == None and result4 == None and result5 == None:
                        f2.write(i)
                    elif result3 !=None:
                        content = "%moinp \"{0}.gbw\"\n".format(filename.split(".")[0])
                        f2.write(content)
                    elif result4 !=None:
                        content1 = "%base \"orca_tmp_up\"\n"
                        f2.write(content1)
                    elif  result5 != None:
                        content2 = "%pal nprocs 4 end\n"
                        content3 = "%TDDFT\n" 
                        content4 = "nroots 5\n"
                        content5 = "iroot  {0}\n".format(str(int(state)))
                        content6 = "tda = false\n"
                        content7 = "printlevel 3 end\n" 
                        f2.write(content2)
                        f2.write(content3)                        
                        f2.write(content4)
                        f2.write(content5)
                        f2.write(content6)
                        f2.write(content7)                  
                f1.close()    
                f2.close()
                self.run(temp_input)
                with open(temp_output) as f:
                    state_symbol = [str(i) for i in range(1,self.state_n+1)]
                    data1 = []
                    data2 = []
                    for i in f:
                        result9 = re.search("E\(SCF\)\s+=\s+-?\d+\.\d+\s+Eh",i)
                        result10 = re.search("STATE\s+\d+:\s+E=\s+-?\d+\.\d+\s+au\s+-?\d+\.\d+\s+eV\s+-?\d+\.\d+\s+cm\*\*-?\d+\s+\<\w+\*\*\d+\>\s+=\s+-?\d+\.\d+",i)
                        if result9 !=None:
                            SCF_Done_energy = float(result9.group().split()[2])
                            data1.append(SCF_Done_energy)
                        elif result10 != None:
                            excited_energy = float(result10.group().split()[3])              
                            data2.append(excited_energy)
                    data_3 = data1 + [(i+data1[0]) for i in data2]
                    energy_dict = dict(zip(state_symbol,data_3[0:self.state_n]))
                    
                    prefix = temp_input.split(".")[0]
                    self.remove_file(prefix)
                    os.remove("orca_tmp_up_property.txt")
                    os.remove("orca_tmp_up.cis")
                    os.remove("orca_tmp_up.engrad")
                    os.remove("orca_tmp_up.densities")
                    os.remove("orca_tmp_up.gbw")
                    if state:
                        return energy_dict[str(int(state))]
                    elif state == False:
                        return energy_dict

