
#!/usr/bin/env python3

import os 
import time 
import random
import numpy as np
from interface import *
from functools import partial
print=partial(print, flush=True)


class surface_hopping():
    
    def __init__(self,program_name,work_dir,init_input,init_output,init_momentum,atom_n,total_time,state_n,step_time,hopping_threshold_value):
        self.program_name = program_name
        self.work_dir = work_dir
        self.init_input = init_input 
        self.init_output = init_output
        self.init_momentum = init_momentum
        self.atom_n = atom_n
        self.total_time = total_time
        self.state_n = state_n
        self.step_time = step_time
        self.hopping_threshold_value = hopping_threshold_value
        self.unit = Unit()
        self.periodic_table = self.unit.periodic_table
        self.au_step_time = self.step_time/self.unit.au_to_fs
        if self.program_name == "MOLPRO":
            self.md = MOLPRO(init_momentum = self.init_momentum,init_input = self.init_input,atom_n = self.atom_n,state_n = self.state_n)
        elif self.program_name == "GAUSSIAN": 
            self.md = GAUSSINA(init_momentum = self.init_momentum,init_input = self.init_input,atom_n = self.atom_n,state_n = self.state_n)
        elif self.program_name == "ORCA":
            self.md = ORCA(init_momentum = self.init_momentum,init_input = self.init_input,atom_n = self.atom_n,state_n = self.state_n)
        self.atom_symbol = self.md.atom_symbol()
 
    #save file
    def save_file(self,filename,content,step,time):
        """_summary_
            this function for save file
        Args:
            filename (_str_): _filename_
        """
        if type(content) == type([]):
            with open(filename,"a+") as f:
                #step time total-energy,elec-energy,kinetic 
                step = ''.join(format(step, '>.2f'))
                time = ''.join(format(time, '>10.2f'))
                lines = step + time
                for i in range(len(content)):
                    energy_data = ''.join(format(content[i], '>20.10f'))
                    lines += energy_data
                f.write(lines + "\n")   
        else:
            with open(filename,"a+") as f:
                f.write(str(self.atom_n))
                f.write("\nStep={0}\tTime={1}\n".format(step,time))
                for i in range(self.atom_n):
                    symbol = ''.join(format(self.md.atom_symbol()[i]))
                    coordinate_1 = ''.join(format(content[i][0], '>20.10f'))
                    coordinate_2 = ''.join(format(content[i][1], '>20.10f'))
                    coordinate_3 = ''.join(format(content[i][2], '>20.10f'))
                    f.write(symbol + coordinate_1 + coordinate_2 + coordinate_3 + "\n")


    def recal(self,input_file,output_file,old_input_file,Q,P,G):
        step_time = self.step_time
        prefix = input_file.split(".")[0]
        for i in range(int(self.step_time/0.1)):
            self.md.remove_file(prefix)
            step_time -= 0.1
            #recalculate Q for step_time -0.1
            Q = self.cal_C(Q,P,G,step_time)
            #replace coordinate
            self.md.replace_coordinate(old_input_file,input_file,Q)
            #run molpro with current input
            self.md.run(input_file)
            #check output
            check_result = self.md.check_out(output_file)
            if step_time > 0.1:
                if check_result != None:
                    print("Back {0}(fs) recalculation is convergence.".format((self.step_time - step_time),'>.2f'))
                    return Q,step_time
            else:
                print("Current can't convergence,program exit!")
                os._exit(0)

    #Calculate coordinate
    def cal_C(self,X,P,G,step_time):
        """
        parameter X:coordinate
        parameter G:gradient
        parameter P:velocity
        X_n+1 = X_n + V_n * Δt +1/2 * a_n * Δt
        """
        au_step_time = step_time/self.unit.au_to_fs
        new_X = np.zeros((self.atom_n,3))
        for i in range(self.atom_n):
            atom_m = self.md.atom_m()[i]
            new_X[i] = X[i]+(P[i]/atom_m)*au_step_time-(0.5*G[i]/atom_m)*au_step_time**2
        return new_X
    
    #Calculate momentum
    def cal_P(self,O_G,G,O_P,step_time):
        """
        parameter O_G:old gradient
        parameter G:new gradient
        parameter O_V:old velocity
        """
        au_step_time = step_time/self.unit.au_to_fs
        new_P = np.zeros((self.atom_n,3))
        for i in range(self.atom_n):
            new_P[i] = O_P[i]-0.5*(O_G[i] + G[i])*au_step_time
        return new_P

    #Calculate Kinetic energy
    def cal_Kin(self,P):
        """
        calculate kinetic enerngy of P
        """

        kin_energy = np.zeros((self.atom_n,3))
        for i in range(self.atom_n):
            atom_m = self.md.atom_m()[i]
            kin_energy[i] = np.power(P[i],2)/(2*atom_m)
        kin_energy = np.sum(kin_energy)
        return kin_energy
        
    #Calculate hopping factor
    def cal_h_f(self,h_s,q1,q2,q3,P,V_u,V_d,g1_u,g2_u,g3_u,g1_d,g2_d,g3_d):
        """
        zhu-nakamura method
        h_s:hopping style.
        g1_u,g3_u,g1_d,g3_d.
        q1,q2,q3:coordinate of n,n-1,n-2.
        V_u,V_d:potential energy,'u' is up,'d'is down.
        """
        #Calculate F1 and F2
        #fiting grdient just be used to calculate hopping probability
        if (q3-q1).all()  <= 0.00005:
            F1 = np.zeros((self.atom_n,3))
            F2 = np.zeros((self.atom_n,3))
        else:
            F1 = np.zeros((self.atom_n,3))
            F2 = np.zeros((self.atom_n,3))
            if h_s == 'U-D':
                for i in range(self.atom_n):
                    F1[i] = -1*((g3_d[i]*(q2[i]-q1[i])-g1_u[i]*(q2[i]-q3[i]))/(q3[i]-q1[i]))
                    F2[i] = -1*((g3_u[i]*(q2[i]-q1[i])-g1_d[i]*(q2[i]-q3[i]))/(q3[i]-q1[i]))
            elif h_s == 'D-U':
                for i in range(self.atom_n):
                    F1[i] = -1*((g3_u[i]*(q2[i]-q1[i])-g1_d[i]*(q2[i]-q3[i]))/(q3[i]-q1[i]))
                    F2[i] = -1*((g3_d[i]*(q2[i]-q1[i])-g1_u[i]*(q2[i]-q3[i]))/(q3[i]-q1[i]))
        
        #Calculate sum of ((F1-F2)^2)/m
        data_1 = np.zeros((self.atom_n,3))
        for i in range(self.atom_n):
            atom_m  = self.md.atom_m()[i]
            data_1[i] = np.power((F1[i]-F2[i]),2)/atom_m
        data_1 = np.sum(data_1)

        #Calculate (F2-F1)/sqrt(m)
        data_2 = np.zeros((self.atom_n,3))
        for i in range(self.atom_n):
            atom_m  = self.md.atom_m()[i]
            data_2[i] = (F2[i]-F1[i])/np.sqrt(atom_m)

        #Calculate F1*F2
        data_3 = np.zeros((self.atom_n,3))
        for i in range(self.atom_n):
                data_3 [i]  = F1[i] * F2[i]
        data_3 = np.sum(data_3)

        #Calculate P direction factor(si)
        #Split P into parallel and vertical direction
        data_4 = np.zeros((self.atom_n,3)) #normalized P direction factor (Si)
        for i in range(self.atom_n):
            data_4[i] = data_2[i]/np.sqrt(data_1)

        #Calculate normalized P factor(ni)
        data_5 = np.zeros((self.atom_n,3))
        for i in range(self.atom_n):
            data_5[i] = data_4[i]/np.linalg.norm(data_4[i]) #np.sum((np.sqrt(np.power(data_4[i],2))))
        
        #Split P in vertical and parallel direction
        P_parallel = np.zeros((self.atom_n,3))
        P_vertical = np.zeros((self.atom_n,3))
        for i in range(self.atom_n):
            P_parallel[i] = np.dot(P[i],data_5[i])*data_5[i] #np.dot(pi,ni)*ni
            P_vertical[i] = P[i] - P_parallel[i]

        #Calculate kinetic energy in parallel and vertical direction
        kin_parallel = self.cal_Kin(P_parallel) #kinetic energy(ev) for parallel 
        kin_vertical =self.cal_Kin(P_vertical) #kinetic energy(ev) for vertical

        #Calculate E_t,E_x,V_x
        E_t = V_u + kin_parallel #potential energy + kinetic enenrgy
        E_x = (V_u + V_d)/2
        V_x = (V_u - V_d)/2


        #Calculate a^2 and b^2
        aa = data_1/(16 * V_x**3)
        bb = (E_t - E_x)/(2 * V_x)

        #Calculate scale factor for V adjustment
        if h_s == "U-D":
            k = np.sqrt(1+(V_x*2)/kin_parallel)
        elif h_s == "D-U":
            k = np.sqrt(1-(V_x*2)/kin_parallel)
            
        #Adjust P in parallel direction after hopping
        new_P_parallel = P_parallel * k
        #Adjust P after hopping
        new_P = new_P_parallel + P_vertical

        if data_3 > 0:
            if aa >= 1000:
                hop_p = 1
            elif aa < 0.001:
                hop_p =0
            else:
                hop_p = np.exp((-1 * self.unit.pi/(4 * np.sqrt(aa))) * np.sqrt(2/(bb + np.sqrt(np.power(bb,2) + 1))))
                print ("The conical intersection is same sign slope.\n")
        if data_3 < 0:
            if aa >= 1000:
                hop_p = 1
            elif aa < 0.001:
                hop_p =0
            else:
                print("The conical intersection is opposite sign slope.\n")
                hop_p = np.exp((-1 * self.unit.pi/(4 * np.sqrt(aa))) * np.sqrt(2/(bb + np.sqrt(abs(np.power(bb,2) - 1)))))
    
        #print detils about hopping point
        #self.md.printf(init_P,'initial Momentun',loop,loop*self.step_time)
        self.md.printf(q1,'q1',"xxx","xxx")
        self.md.printf(g1_u,'g1_u',"xxx","xxx")
        self.md.printf(g1_d,'g1_d',"xxx","xxx")
        self.md.printf(q3,'q3',"xxx","xxx")
        self.md.printf(g3_u,'g3_u',"xxx","xxx")
        self.md.printf(g3_d,'g3_d',"xxx","xxx")
        self.md.printf(q2,'q2',"xxx","xxx")
        self.md.printf(g2_u,'g2_u',"xxx","xxx")
        self.md.printf(g2_d,'g2_d',"xxx","xxx")
        self.md.printf(data_2,'(F1-F2)/sqrt(m)',"xxx","xxx")
        self.md.printf(data_4,'P direction factor(si)',"xxx","xxx")
        self.md.printf(data_5,'normalized P factor(ni)',"xxx","xxx")
        self.md.printf(P_parallel,'P_parallel',"xxx","xxx")
        self.md.printf(P_vertical,'P_vertical',"xxx","xxx")
        self.md.printf(new_P_parallel,'Adjust P in parallel direction after hopping',"xxx","xxx")
        self.md.printf(P,'Adjust P befor hopping',"xxx","xxx")
        self.md.printf(new_P,'Adjust P after hopping',"xxx","xxx")
        print('\nsum of ((F1-F2)^2)/m',data_1)
        print("sum of F1*F2:",data_3)
        print("kin_parallel(ev):",kin_parallel*self.unit.au_to_ev)
        print("kin_vertical(ev):",kin_vertical*self.unit.au_to_ev)
        print("V_u:",V_u)
        print("V_d:",V_d)
        print("E_t",E_t)
        print("E_x",E_x)
        print("V_x",V_x)
        print("a^2:",aa)
        print("b^2:",bb)
        print("hop_p",hop_p)
        print("k:",k)
        return hop_p,new_P

    #check hopping
    def  check_energy_gap(self,energy_states,filename_q2,loop):
        
        """
        filename_q1:loop-2 output
        filename_q2:loop-1 output
        filename_q3:loop output
        we need to check shape of concial intersection(CI)
        """
        
        c_state = float(self.md.state(filename_q2))
        if int(c_state) == 1:#if current state is lowest such as S0, S0->S1
            energy_current_state_q1 = energy_states[loop-2][int(c_state-1)]
            energy_current_state_q2 = energy_states[loop-1][int(c_state-1)]
            energy_current_state_q3 = energy_states[loop][int(c_state-1)]
            energy_up_state_q1 = energy_states[loop-2][int(c_state)]
            energy_up_state_q2 = energy_states[loop-1][int(c_state)]
            energy_up_state_q3 = energy_states[loop][int(c_state)]
            enegry_gap_q1 = abs(energy_up_state_q1 -energy_current_state_q1)*self.unit.au_to_ev
            enegry_gap_q2 = abs(energy_up_state_q2 -energy_current_state_q2)*self.unit.au_to_ev
            enegry_gap_q3 = abs(energy_up_state_q3 -energy_current_state_q3)*self.unit.au_to_ev
            print("Enegry_gap_q1(ev):{0}\tEnegry_gap_q2(ev):{1}\tEnegry_gap_q3(ev):{2}".format(enegry_gap_q1,enegry_gap_q2,enegry_gap_q3))
            if enegry_gap_q1 > enegry_gap_q2 < enegry_gap_q3 and enegry_gap_q2 < self.hopping_threshold_value:
                return 'D-U',energy_up_state_q2,energy_current_state_q2,c_state+1,c_state
            else:
                return None,None,None,None,None
            
        if 1 < int(c_state) < self.state_n:#if current is middle state such as S1,S0<-S1->S2.
            energy_down_state_q1 = energy_states[loop-2][int(c_state-2)]
            energy_down_state_q2 = energy_states[loop-1][int(c_state-2)]
            energy_down_state_q3 = energy_states[loop][int(c_state-2)]
            energy_current_state_q1 = energy_states[loop-2][int(c_state-1)]
            energy_current_state_q2 = energy_states[loop-1][int(c_state-1)]
            energy_current_state_q3 = energy_states[loop][int(c_state-1)]
            energy_up_state_q1 = energy_states[loop-2][int(c_state)]
            energy_up_state_q2 = energy_states[loop-1][int(c_state)]
            energy_up_state_q3 = energy_states[loop][int(c_state)]
            enegry_gap_c_d_state_q1 = abs(energy_current_state_q1 - energy_down_state_q1)*self.unit.au_to_ev
            enegry_gap_c_d_state_q2 = abs(energy_current_state_q2 - energy_down_state_q2)*self.unit.au_to_ev
            enegry_gap_c_d_state_q3 = abs(energy_current_state_q3 - energy_down_state_q3)*self.unit.au_to_ev
            enegry_gap_c_u_state_q1 = abs(energy_up_state_q1 - energy_current_state_q1)*self.unit.au_to_ev
            enegry_gap_c_u_state_q2 = abs(energy_up_state_q2 - energy_current_state_q2)*self.unit.au_to_ev
            enegry_gap_c_u_state_q3 = abs(energy_up_state_q3 - energy_current_state_q3)*self.unit.au_to_ev
            print("Enegry_gap_c_u_state_q1(ev):{0}\tEnegry_gap_c_u_state_q2(ev):{1}\tEnegry_gap_c_u_state_q3(ev):{2}".format(enegry_gap_c_u_state_q1,enegry_gap_c_u_state_q2,enegry_gap_c_u_state_q3))
            print("Enegry_gap_c_d_state_q1(ev):{0}\tEnegry_gap_c_d_state_q2(ev):{1}\tEnegry_gap_c_d_state_q3(ev):{2}".format(enegry_gap_c_d_state_q1,enegry_gap_c_d_state_q2,enegry_gap_c_d_state_q3))
            if enegry_gap_c_d_state_q1 > enegry_gap_c_d_state_q2 < enegry_gap_c_d_state_q3 and enegry_gap_c_d_state_q2 < self.hopping_threshold_value:
                return 'U-D',energy_current_state_q2,energy_down_state_q2,c_state,c_state-1
            elif enegry_gap_c_u_state_q1 > enegry_gap_c_d_state_q2 < enegry_gap_c_u_state_q3 and enegry_gap_c_u_state_q2 < self.hopping_threshold_value:
                return 'D-U',energy_up_state_q2,energy_current_state_q2,c_state+1,c_state
            else:
                return None,None,None,None,None
            
        if int(c_state) == self.state_n:#if current is lowest state such as S2,S2->S1
            energy_current_state_q1 = energy_states[loop-2][int(c_state-1)]
            energy_current_state_q2 = energy_states[loop-1][int(c_state-1)]
            energy_current_state_q3 = energy_states[loop][int(c_state-1)]
            energy_down_state_q1 = energy_states[loop-2][int(c_state-2)]
            energy_down_state_q2 = energy_states[loop-1][int(c_state-2)]
            energy_down_state_q3 = energy_states[loop][int(c_state-2)]
            enegry_gap_q1 = abs(energy_current_state_q1 - energy_down_state_q1)*self.unit.au_to_ev
            enegry_gap_q2 = abs(energy_current_state_q2 - energy_down_state_q2)*self.unit.au_to_ev
            enegry_gap_q3 = abs(energy_current_state_q3 - energy_down_state_q3)*self.unit.au_to_ev
            print("Enegry_gap_q1(ev):{0}\tEnegry_gap_q2(ev):{1}\tEnegry_gap_q3(ev):{2}".format(enegry_gap_q1,enegry_gap_q2,enegry_gap_q3))
            if enegry_gap_q1 > enegry_gap_q2 < enegry_gap_q3 and enegry_gap_q2 < self.hopping_threshold_value:
                return 'U-D',energy_current_state_q2,energy_down_state_q2,c_state,c_state-1
            else:
                return None,None,None,None,None
            
    def hopping(self,h_s,V_u,V_d,state_u,state_d,g,q,p,potential_energy,kinetic_energy,total_energy,energy_states,input_list,output_list,loop,time_list):
            """
            if "hopping" is not "None",we need get the P of n-1, the G of n,n-1, n-2 at different state
            calculate hopping factor
            h_s:hopping style.
            g1_u,g3_u,g1_d,g3_d.
            q1,q2,q3:coordinate of n,n-1,n-2.
            V_u,V_d:potential energy,'u' is up,'d'is down.
            input_list:the list of input
            output_list:the list of output
            """
            step_time = time_list[-1]
            hopping_flag = [0]
            if h_s == 'U-D' and loop - hopping_flag[-1] > 3:
                print("hopping event!!!")
                #calculate g1_u,g1_d,g3_u,g3_d
                g1_u = self.md.gradient(output_list[loop-2])
                g2_u = self.md.gradient(output_list[loop-1])
                g3_u = self.md.gradient(output_list[loop])
                prefix_q1 = input_list[loop-2].split(".")[0]+"_d"
                prefix_q2 = input_list[loop-1].split(".")[0]+"_d"
                prefix_q3 = input_list[loop].split(".")[0]+"_d"
                q1_temp_in,q1_temp_out = self.md.gen_filename(prefix_q1)
                q2_temp_in,q2_temp_out = self.md.gen_filename(prefix_q2)
                q3_temp_in,q3_temp_out = self.md.gen_filename(prefix_q3)
                self.md.replace_wfu_filename(input_list[loop-2],"temp-file-q1")
                self.md.replace_wfu_filename(input_list[loop-1],"temp-file-q2")
                self.md.replace_wfu_filename(input_list[loop],"temp-file-q3")
                self.md.replace_state("temp-file-q1",q1_temp_in,state_d)
                self.md.replace_state("temp-file-q2",q2_temp_in,state_d)
                self.md.replace_state("temp-file-q3",q3_temp_in,state_d)
                #replace wfu filename.
                self.md.run(q1_temp_in)
                self.md.run(q2_temp_in)
                self.md.run(q3_temp_in)
                g1_d = self.md.gradient(q1_temp_out)
                g2_d = self.md.gradient(q2_temp_out)
                g3_d = self.md.gradient(q3_temp_out)
                self.md.remove_file(prefix_q1)
                self.md.remove_file(prefix_q2)
                self.md.remove_file(prefix_q3)
                os.remove("temp-file-q1")
                os.remove("temp-file-q2")
                os.remove("temp-file-q3")
                hopping_p,P_q2= self.cal_h_f(h_s,q[loop-2],q[loop-1],q[loop],p[loop-1],V_u,V_d,g1_u,g2_u,g3_u,g1_d,g2_d,g3_d)
                random_number = random.random()
                print("hopping_p is:{0}\trandom_number:{1}".format(hopping_p,random_number))
                #the P is new P for q2
                if hopping_p >= random_number:#if hopping is successful,recalculate current step(loop)
                    old_Q = self.md.coordinate(input_list[loop-1])
                    old_P = P_q2
                    old_G = g2_d #g2_d for old_G
                    Q = self.cal_C(old_Q,old_P,old_G,step_time)
                    os.remove(input_list[loop])
                    os.remove(output_list[loop])
                    self.md.replace_coordinate(self.init_input,"temp-file",Q)
                    self.md.replace_state("temp-file",input_list[loop],state_d)
                    self.md.run(input_list[loop])
                    check_result = self.md.check_out(output_list[loop])
                    if check_result != None:#for convergence
                        print(check_result)
                        print("Ab initial calculation is convergence!")
                        G = self.md.gradient(output_list[loop])
                        P = self.cal_P(old_G,G,old_P,step_time)
                        time_list.append(self.step_time)
                    elif check_result == None:#for misconvergence
                        print("Ab initial calculation is misconvergence!")
                        Q,back_step_time = self.recal(input_list[loop],output_list[loop],"temp-file",old_Q,old_P,old_G)
                        G = self.md.gradient(output_list[loop])
                        P= self.cal_P(g[loop-1],G,p[loop-1],back_step_time)
                        step_time = back_step_time
                        time_list.append(1 - back_step_time)
                    kin_energy = self.cal_Kin(P)
                    current_state = float(self.md.state(output_list[loop]))
                    energy_current_state = self.md.energy(output_list[loop],current_state)
                    q[loop] = Q
                    g[loop] = G
                    g[loop-1] = old_G
                    p[loop] = P
                    potential_energy[loop] = energy_current_state
                    kinetic_energy[loop] = kin_energy
                    total_energy[loop] = kin_energy + energy_current_state
                    energy_states[loop] = list(self.md.energy(output_list[loop],False).values())
                    hopping_flag.append(loop)
                    os.remove("temp-file")
                    self.md.printf(Q*self.unit.au_to_ang,'re-calculate q3 coordinate',loop+1,(loop*self.step_time + step_time))
                    self.md.printf(G,'re-calculate q3 Gradient',loop+1,(loop*self.step_time + step_time))
                    self.md.printf(P,'re-calculate q3 Momentun',loop+1,(loop*self.step_time + step_time))
                    print("step {0} time {1} hopping event: {2} -> {3}".format(loop+1,(loop*self.step_time + step_time),state_u,state_d))
                    with open("hopping.log","a+") as f:
                        f.write("step {0} time {1} hopping event: {2} -> {3}\n".format(loop+1,(loop*self.step_time + step_time),state_u,state_d))
                    print("Hopping successful!")
                else:
                    print("Hopping failure!")

            elif h_s == 'D-U' and loop - hopping_flag[-1] > 3:
                g1_d = self.md.gradient(output_list[loop-2])
                g2_d = self.md.gradient(output_list[loop-1])
                g3_d = self.md.gradient(output_list[loop])
                prefix_q1 = input_list[loop-2].split(".")[0]+"_u"
                prefix_q2 = input_list[loop-1].split(".")[0]+"_u"
                prefix_q3 = input_list[loop].split(".")[0]+"_u"
                q1_temp_in,q1_temp_out = self.md.gen_filename(prefix_q1)
                q2_temp_in,q2_temp_out = self.md.gen_filename(prefix_q2)
                q3_temp_in,q3_temp_out = self.md.gen_filename(prefix_q3)
                self.md.replace_wfu_filename(input_list[loop-2],"temp-file-q1")
                self.md.replace_wfu_filename(input_list[loop-1],"temp-file-q2")
                self.md.replace_wfu_filename(input_list[loop],"temp-file-q3")
                self.md.replace_state("temp-file-q1",q1_temp_in,state_u)
                self.md.replace_state("temp-file-q2",q2_temp_in,state_u)
                self.md.replace_state("temp-file-q3",q3_temp_in,state_u)
                #replace wfu filename.
                self.md.run(q1_temp_in)
                self.md.run(q2_temp_in)
                self.md.run(q3_temp_in)
                g1_u = self.md.gradient(q1_temp_out)
                g2_u = self.md.gradient(q2_temp_out)
                g3_u = self.md.gradient(q3_temp_out)
                self.md.remove_file(prefix_q1)
                self.md.remove_file(prefix_q2)
                self.md.remove_file(prefix_q3)
                os.remove("temp-file-q1")
                os.remove("temp-file-q2")
                os.remove("temp-file-q3")
                random_number = random.random()
                #calculate hopping factor
                #hopping_p: hopping probability
                hopping_p,P_q2 = self.cal_h_f(h_s,q[loop-2],q[loop-1],q[loop],p[loop-1],V_u,V_d,g1_u,g2_u,g3_u,g1_d,g2_d,g3_d)
                print("hopping_p is:{0}\trandom_number:{1}".format(hopping_p,random_number))
                if hopping_p >= random_number:
                    old_Q = self.md.coordinate(input_list[loop-1])
                    old_P = P_q2
                    old_G = g2_u #g2_d for old_G
                    Q = self.cal_C(old_Q,old_P,old_G,step_time)
                    os.remove(input_list[loop])
                    os.remove(output_list[loop])
                    self.md.replace_coordinate(self.init_input,"temp-file",Q)
                    self.md.replace_state("temp-file",input_list[loop],state_d)
                    self.md.run(input_list[loop])
                    check_result = self.md.check_out(output_list[loop])
                    if check_result != None:#for convergence
                        print(check_result)
                        print("Ab initial calculation is convergence!")
                        G = self.md.gradient(output_list[loop])
                        P = self.cal_P(old_G,G,old_P,step_time)
                        time_list.append(self.step_time)
                    elif check_result == None:#for misconvergence
                        print("Ab initial calculation is misconvergence!")
                        Q,back_step_time = self.recal(input_list[loop],output_list[loop],"temp-file",old_Q,old_P,old_G)
                        G = self.md.gradient(output_list[loop])
                        P= self.cal_P(old_G,G,old_P,back_step_time)
                        step_time = back_step_time
                        time_list.append(1 - back_step_time)
                    kin_energy = self.cal_Kin(P)
                    current_state = float(self.md.state(output_list[loop]))
                    energy_current_state = self.md.energy(output_list[loop],current_state)
                    q[loop] = Q
                    g[loop] = G
                    g[loop-1] = old_G
                    p[loop] = P
                    potential_energy[loop] = energy_current_state
                    kinetic_energy[loop] = kin_energy
                    total_energy[loop] = kin_energy + energy_current_state
                    energy_states[loop] = list(self.md.energy(output_list[loop],False).values())
                    hopping_flag.append(loop)
                    os.remove("temp-file")
                    self.md.printf(Q*self.unit.au_to_ang,'re-calculate q3 coordinate',loop+1,(loop*self.step_time + step_time))
                    self.md.printf(G,'re-calculate q3 Gradient',loop+1,(loop*self.step_time + step_time))
                    self.md.printf(P,'re-calculate q3 Momentun',loop+1,(loop*self.step_time + step_time))
                    print("step {0} time {1} hopping event: {2} -> {3}".format(loop+1,(loop*self.step_time + step_time),state_u,state_d))
                    with open("hopping.log","a+") as f:
                        f.write("step {0} time {1} hopping event: {2} -> {3}\n".format(loop+1,(loop*self.step_time + step_time),state_d,state_u))
                    print("Hopping successful!")
                else:
                    print("Hopping failure!")
            else:
                print("Hopping step between two hopping point is small,Hopping failure!")
                                        

    def on_the_fly(self):
        loop = 0
        current_time  = 0
        #1.check file:input,momentum
        print("Start Molecular Dynamics Simulation")
        print("Localtime:",time.asctime(time.localtime(time.time())))
        start_time = time.time()
        print("Current work dir is:",self.work_dir)
        print("Check initial molpro input file")
        self.md.check(self.init_input)
        print("Check inital momentum file")
        self.md.check(self.init_momentum)
        print("au_step_time(a.u.):",self.au_step_time)
        print("atom symbol:",set(self.md.atom_symbol()))
        print("atom mass(a.u.):", np.unique(self.md.atom_m().reshape(-1)))
        q = []
        g = []
        p = []
        time_list = []
        energy_states = []
        potential_energy = []
        kinetic_energy = []
        total_energy = []
        input_list = []
        output_list = []

        #loop
        while loop < self.total_time/self.step_time:
            """
            coordinate matrix
            gradient matrix
            mommentum matrix

            2.run molpro
            3.calculate coordinate
            4.get momentum
            5.update coordinate: calculate and replace new coordinate
            6.save q1,q2,q3
            """
            
            if loop == 0:
                #======================initial step=====================
                # calculate initial iniput by molpro
                self.md.run(self.init_input)
                #check initial output
                check_result = self.md.check_out(self.init_output)
                if check_result != None:#for convergence
                    print(check_result)
                    print("Ab initial calculation at inital step is convergence!")
                else:#for misconvergence
                    print("Ab initial calculation at inital step is misconvergence!")
                    print("program is exit!")
                    os._exit(0)
                #initial coordinate
                init_Q = self.md.coordinate(self.init_input)
                #initial gradient
                init_G = self.md.gradient(self.init_output)
                #initial momentum
                init_P = self.md.momentum(self.init_momentum)
                #print Q P G
                self.md.printf(init_Q*self.unit.au_to_ang,'initial Coordinate',loop,loop*self.step_time)
                self.md.printf(init_G,'initial Gradient',loop,loop*self.step_time)
                self.md.printf(init_P,'initial Momentun',loop,loop*self.step_time)
                #print summary information
                print('\n{0:*^63}\n'.format(" summary information "))
                init_kin = self.cal_Kin(init_P)
                spin = self.md.spin(self.init_output)
                total_P_axis = np.sum(init_P,0)
                current_state = float(self.md.state(self.init_output))
                print("inital input is:",self.init_output)
                print("initial output is:",self.init_output)
                print("Total state is:",self.state_n)
                energy_current_state = self.md.energy(self.init_output,current_state)
                print("Step {0} time {1}(fs) state: {2} spin: {3} energy(a.u.): {4}".format(loop,loop*self.step_time,current_state,spin,energy_current_state))
                print("Step {0} time {1}(fs) kinetic energy(a.u.): {2}".format(loop,loop*self.step_time,init_kin))
                print("Step {0} time {1}(fs) Total energy (a.u.): {2}".format(loop,loop*self.step_time,init_kin + energy_current_state))
                print("total momentum at x axis:",total_P_axis[0])
                print("total momentum at y axis:",total_P_axis[1])
                print("total momentum ay z axis:",total_P_axis[2])
                #output the Q,G,P,energy
                self.save_file("coordinate.xyz",init_Q*self.unit.au_to_ang,loop,loop*self.step_time)
                self.save_file("grdient.log",init_G,loop,loop*self.step_time)
                self.save_file("momentum.log",init_P,loop,loop*self.step_time)
                self.save_file("energy.log",[init_kin + energy_current_state,energy_current_state,init_kin] + list(self.md.energy(self.init_output,False).values()),loop,loop*self.step_time)
                time_list.append(self.step_time)
                #======================initial step=====================
                
                
                step_time = time_list[-1]
                #calculate next step coordinate
                Q = self.cal_C(init_Q,init_P,init_G,step_time)
                #generate the name of out and new input
                prefix = self.init_input.split(".")[0]+"_"+str(loop +1)
                input_file,output_file = self.md.gen_filename(prefix)
                #replace coordinate
                self.md.replace_coordinate(self.init_input,input_file,Q)
                #run molpro with current input
                self.md.run(input_file)
                #check output
                check_result = self.md.check_out(output_file)
                if check_result != None:#for convergence
                    print(check_result)
                    print("Ab initial calculation is convergence!")
                    #get old gradient
                    G = self.md.gradient(output_file)
                    #get curretn momentun
                    P= self.cal_P(init_G,G,init_P,step_time)
                    time_list.append(self.step_time)
                elif check_result == None:#for misconvergence
                    print("Ab initial calculation is misconvergence!")
                    Q,back_step_time = self.recal(input_file,output_file,self.init_input,init_Q,init_P,init_G)
                    #get current gradient
                    G = self.md.gradient(output_file)
                    #get curretn momentun
                    P= self.cal_P(init_G,G,init_P,back_step_time)
                    step_time = back_step_time
                    time_list.append(1 - back_step_time)
                current_time += step_time
                q.append(Q)
                g.append(G)
                p.append(P)
                input_list.append(input_file)
                output_list.append(output_file)
                #print Q,G,P
                self.md.printf(Q*self.unit.au_to_ang,'Coordinate',str(loop + 1),current_time)
                self.md.printf(G,'Gradient',str(loop + 1),current_time)
                self.md.printf(P,'Momentun',str(loop + 1),current_time)
                #print summary information
                print('\n{0:*^63}\n'.format(" summary information "))
                spin = self.md.spin(output_file)
                kin_energy = self.cal_Kin(P)
                total_P_axis = np.sum(P,0)
                current_state = float(self.md.state(output_file))
                print("Current input is:",input_file)
                print("Current output is:",output_file)
                print("Total state is:",self.state_n)
                energy_current_state = self.md.energy(output_file,current_state)
                print("Step {0} time {1}(fs) state: {2} spin: {3} energy(a.u.): {4}".format((loop + 1),current_time,current_state,spin,energy_current_state))
                print("Step {0} time {1}(fs) kinetic energy(a.u.): {2}".format((loop + 1),current_time,kin_energy))
                print("Step {0} time {1}(fs) Total energy (a.u.): {2}".format((loop + 1),current_time,kin_energy+energy_current_state))
                print("total momentum at x axis:",total_P_axis[0])
                print("total momentum at y axis:",total_P_axis[1])
                print("total momentum ay z axis:",total_P_axis[2])
                potential_energy.append(energy_current_state)
                kinetic_energy.append(kin_energy)
                total_energy.append(kin_energy + energy_current_state)
                energy_states.append(list(self.md.energy(output_file,False).values()))
                #output the Q,G,P,energy
                self.save_file("coordinate.xyz",q[loop]*self.unit.au_to_ang,loop + 1,current_time)
                self.save_file("grdient.log",g[loop],loop + 1,current_time)
                self.save_file("momentum.log",p[loop],loop + 1,current_time)
                self.save_file("energy.log",[total_energy[loop],potential_energy[loop],kinetic_energy[loop]]+ energy_states[loop],loop + 1,current_time)
                #loop
                loop += 1
            else:
                step_time = time_list[-1]
                #calculated current coordinate
                Q = self.cal_C(q[loop-1],p[loop-1],g[loop-1],step_time)
                #generate file name for current loop
                prefix = self.init_input.split(".")[0]+"_"+str(loop +1)
                input_file,output_file = self.md.gen_filename(prefix)
                #replace coordinate
                self.md.replace_coordinate(input_list[loop-1],input_file,Q)
                #run molpro with current input
                self.md.run(input_file)
                #check output
                check_result = self.md.check_out(output_file)
                if check_result != None:#for convergence
                    print(check_result)
                    print("Ab initial calculation is convergence!")
                    #get current gradient
                    G = self.md.gradient(output_file)
                    #get curretn momentun
                    P= self.cal_P(g[loop-1],G,p[loop-1],step_time)
                    time_list.append(self.step_time)
                elif check_result == None:#for misconvergence
                    print("Ab initial calculation is misconvergence!")
                    Q,back_step_time = self.recal(input_file,output_file,input_list[loop-1],q[loop-1],p[loop-1],g[loop-1])
                    #get current gradient
                    G = self.md.gradient(output_file)
                    #get curretn momentun
                    P= self.cal_P(g[loop-1],G,p[loop-1],back_step_time)
                    step_time = back_step_time
                    time_list.append(1 - back_step_time)
                current_time += step_time
                #print Q,G,P
                self.md.printf(Q*self.unit.au_to_ang,'Coordinate',(loop + 1),current_time)
                self.md.printf(G,'Gradient',(loop + 1),current_time)
                self.md.printf(P,'Momentun',(loop + 1),current_time)
                q.append(Q)
                g.append(G)
                p.append(P)
                input_list.append(input_file)
                output_list.append(output_file)
                #print summary information
                print('\n{0:*^63}\n'.format(" summary information "))
                spin = self.md.spin(output_file)
                kin_energy = self.cal_Kin(P)
                total_P_axis = np.sum(P,0)
                current_state = float(self.md.state(output_file))
                print("Current input is:",input_file)
                print("Current output is:",output_file)
                print("Total state is:",self.state_n)
                energy_current_state = self.md.energy(output_file,current_state)
                print("Step {0} time {1}(fs) state: {2} spin: {3} energy(a.u.): {4}".format((loop + 1),current_time,current_state,spin,energy_current_state))
                print("Step {0} time {1}(fs) kinetic energy(a.u.): {2}".format((loop + 1),current_time,kin_energy))
                print("Step {0} time {1}(fs) Total energy (a.u.): {2}".format((loop + 1),current_time,kin_energy+energy_current_state))
                print("total momentum at x axis:",total_P_axis[0])
                print("total momentum at y axis:",total_P_axis[1])
                print("total momentum ay z axis:",total_P_axis[2])
                potential_energy.append(energy_current_state)
                kinetic_energy.append(kin_energy)
                total_energy.append(kin_energy + energy_current_state)
                energy_states.append(list(self.md.energy(output_file,False).values()))
                #check back step energy gap between two state
                                
                #hopping
                if loop >=2:
                    print("Check energy gap between two state!")
                    print("Check step {0} time {1}(fs) filename is: {2}".format((loop + 1)-1,current_time,output_list[loop-1]))
                    h_s,V_u,V_d,state_u,state_d = self.check_energy_gap(energy_states,output_list[loop-1],loop)
                    if  h_s !=None:
                        print(h_s,V_u,V_d,state_u,state_d)
                        self.hopping(h_s,V_u,V_d,state_u,state_d,g,q,p,potential_energy,kinetic_energy,total_energy,energy_states,input_list,output_list,loop,time_list)
                if loop>=3:
                    prefix = self.init_input.split(".")[0]+"_"+str(loop-2)
                    self.md.remove_file(prefix)
                #output the Q,G,P,energy
                self.save_file("coordinate.xyz",q[loop]*self.unit.au_to_ang,loop + 1,current_time)
                self.save_file("grdient.log",g[loop],loop + 1,current_time)
                self.save_file("momentum.log",p[loop],loop + 1,current_time)
                self.save_file("energy.log",[total_energy[loop],potential_energy[loop],kinetic_energy[loop]] + energy_states[loop],loop + 1,current_time)
                #loop
                loop += 1
            
        #computional time
        end_time = time.time()
        print("Time consuming:\t{0}(seconds)".format(end_time-start_time))

