# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 10:57:24 2025|| Version:v1
Modified on Tuesday Oct 05 2026||Version:v2
@author: Amit Singh, The University of Alabama, USA||amitdravid@outlook.com, asingh38@crimson.ua.edu
Description: Linear Elastic deformation of 2D sheet under tensile loading, 
            FEM code based on Textbook "First Course on FEM" by Jacob Fish and Ted Belytschko, Chapter:9
About v2: In this version of LFEM-2D plate, instead of applying load-BC, we applied displacement-BC incremently 
Note: 1. In this exercise 4-node rectangular elements are considered
      2. Node numbering in element is done anticlock-wise 1,2,3,4 are in (-1,-1), (1,-1),(1,1),(-1,1) in parent-space of xi and eta, respectively
      3. 
Input: "elements.txt" file, which conatins node number of each elements
     "node_coordinates.txt" file which conatins coordinates of the nodes 
     "configuration.txt" file which contains details about the Material parameters,Boundary-Conditions, Applied displacement, and node of interest
"""
#%% Importing required Libaries/FUnction
import numpy as np
import pandas as pd
from fem_functions_lib import *
import matplotlib.pyplot as plt
#%% Ploting requirment
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
#%% Section-0 Loading configuration file
# This file contains all the parameters required for Linear-FEM simulation
#Note: The variables used in the code should be edited in "configuration.txt" file on the specific line corresponding to that variable
with open("configuration.txt", "r") as f:
    config = []
    config_lines = {4, 6, 10, 12, 14, 16, 20}
    for line_number, line in enumerate(f, start=1):
        if line_number in config_lines:
            text = line.strip()
            if "." in text or "e" in text.lower():
                value = float(text)
            else:
                value = int(text)
            config.append(value)
#%% Section-1 Materials Parameters
## 1.1 Elastic properties
E = config[0] 
nu = config[1]
## 1.2 The D matrix
D = D_matrix(E, nu, mode="plane_stress")
#%% Section-2: Geometry and Guass points
## 2.1 Elements i.e. Connectivity matrix
#Note: Elements numebing shoud be start from 1
elements = pd.read_excel(f"elements_16x20.xlsx", header =None)
elements -= 1 # This is to make sure elements index from 0  not from 1

## 2.2 Node position
nodecords = pd.read_excel(f"nodal_coordinates_16x20.xlsx", header =None)
nel = elements.shape[0] # Number of elements
nnd = nodecords.shape[0] # number of nodes

## 2.3 Plate
plate_thickness = 1
plate_width  = nodecords[0].max() - nodecords[0].min()   # x-extent
plate_height = nodecords[1].max() - nodecords[1].min()   # y-extent
cross_section_area = plate_width * plate_thickness

## 2.4 Elements coordinates and mesh plot
elem_cord = []
for ii in range(nel):
    elm = elements.loc[ii]
    # print(ii)
    node1 =  nodecords.loc[elm[0]]
    node2 =  nodecords.loc[elm[1]]
    node3 =  nodecords.loc[elm[2]]
    node4 =  nodecords.loc[elm[3]]
    elem_cord.append(np.vstack([node1, node2, node3, node4])) # just to save element cord
    
## 2.5 Guass Quadrature point and weights for Quadrilateral elements per element
ngp = 2 # Number of Guass-Qadrature point in each direction i.e. 4 Guass Point
qp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)]) # Quadrature Points
W = np.array([1,1]) # Weights of QP
nne = 4 # Number of Nodes per elements
ndf = 2 # Number of Degree of freedom per node
## 2.6 System of number of equations for entire geometry
neq = nnd*ndf

#%% Section-3: Boundary conditions i.e prescribing Force and Displacements, but here we are just have both BCs as displacement
##-------------- Prescribing Natural BC via displacement -----------------------------
nx, ny = config[2], config[3] # Number of elements for fixed BCs
nnx = nx+1
# Defing the nodes at which force is being applied
nodes_nbc = np.arange(nnd-nnx,nnd)
nbc_df = prescribe_nodal_bc(nodes_nbc,ndf) #nbc_df = nodes for natural BC's degree of freedom (x,y) i.e. 2 = 5*2
nbc_y_df = 2*nodes_nbc +1 # only y perscribed, x-stay free for contrcation (Poisson affect)

##-------------- Prescribing Essential BC via displacement -----------------------------
# Defining nodes at which displacemnt is prescribed
nodes_ebc = np.arange(nnx) 
ebc_df = prescribe_nodal_bc(nodes_ebc,ndf) # ebc_df = nodes for essential BC's degree of freedom (x,y) i.e.2 = 5*2

## The degree of freedom from the nodes which are free from both the BCs
prescribed_df =  np.concatenate([ebc_df, nbc_y_df])
free_df = np.array([d for d in range(neq) if d not in set(prescribed_df)])

## The parent coords of local nodes i.e. corner nodes as per isoparametric mapping, with the same order as the node numbering of element
cord_parent = [(-1,-1),(1,-1),(1,1),(-1,1)]

#%% Section-4: Assemble linear elastic stiffness (K); solving for ke and fe for each element
## 4.1 Body force
b_body = np.zeros([nne*ndf, nel]) # element wise, body force matrix, most of the time zero
## 4.2 Systems of equations for entire geometry
# Initialization for K, d and r
K = np.zeros([neq, neq]) # Node wise stiffness matrix
F0 = np.zeros([neq]) # Node wise force vector
## 4.3 Solving for Kd = F element wise
for el in range(nel):
    cord_e = elem_cord[el] # elemental coordinates, element wise
    bf_e = b_body[:,el] # body force elemnet wise
    Ke, fe = elemental_stiffness(nne, ndf,ngp, cord_e, bf_e, qp,W,D)
    elm_node = elements.loc[el,:]
    index = gather_index(nne, ndf, elm_node)
    K, F0 = assembly(K,F0, Ke,fe, index)

#%% Section-5: Monitored element details
## 5.1
monitor_node = int(nnd - nnx + nx/2)         # node #28 (1-indexed), top-center, index 27
monitor_df_y = 2*monitor_node + 1
# monitor_element = 18         # a representative element (mid-height column)
## 5.2 Finding elements that are touching the "monitor node"
elems_touching_node = []
for el in range(nel):
    node_ids = list(elements.loc[el, :4])
    if monitor_node in node_ids:
        local_i = node_ids.index(monitor_node)
        xi, eta = cord_parent[local_i]
        index = gather_index(nne, ndf,elements.loc[el,:])
        elems_touching_node.append((el, xi, eta, index))
#%% Section-6: Displacement controlled deformation steps
Uy_applied = config[4]     # Displacement applied on top nodes of the body as natural BCs,
nsteps = config[5] # Number steps

disp_hist, force_hist = [], []
global_stress_record, global_strain_record = [], []
U_hist = [] # To store full U at each step
node_syy_hist, node_eyy_hist = [], []
for step in range(1, nsteps+1):
    frac = step/nsteps
    Uy_target = frac*Uy_applied
    U = np.zeros(neq)
    U[nbc_y_df] = Uy_target # Perscribing incrementally the natural BCs to top nodes's y DOF
    U[ebc_df] = 0 #Applied displacement at the perscribed nodes at the bootom nodes
    ## Partitioning K matrix and U and F array, based on free and prescribed DOF and
    K_ff = K[np.ix_(free_df, free_df)] # K_ff = K matrix of free-DOF
    K_fp = K[np.ix_(free_df, prescribed_df)] # K_fp = K matrix of free-DOF and perscribed-DOF
    U_p = U[prescribed_df]
    F_f = F0[free_df]
    F_eff_f = F_f - np.matmul(K_fp, U_p) # Define as Effective free vector
    U_f = np.linalg.solve(K_ff, F_eff_f)
    U[free_df] = U_f
    ## Full internal force KU  = F
    F = np.matmul(K, U)
    ## Reaction force KU  = F_ext +R
    R = -F[ebc_df][1::2].sum()
    ## Saving the records
    disp_hist.append(U[monitor_df_y])
    force_hist.append(R)
    global_stress_record.append(R/cross_section_area)
    global_strain_record.append(U[monitor_df_y]/plate_height)
    U_hist.append(U.copy())
    ##++++++++++++++++ Finding stress-strain at monitor node per displacement increment++++++++++++++
    syy_sum, eyy_sum = 0, 0 # Stress and strain
    for el, xi,eta, index in elems_touching_node:
        u_e = U[index] # node-wise displacement of monitor element
        _, _, strain, stress = field_at_point(xi, eta, elem_cord[el], u_e, D)
        syy_sum += stress[1]
        eyy_sum += strain[1]
    ## Reducing the weight to that node shared adjacent elements
    n_touch = len(elems_touching_node)
    node_syy_hist.append(syy_sum / n_touch)
    node_eyy_hist.append(eyy_sum / n_touch)
U_final = U_hist[-1]
print(f"Elastic Modulus(E)): "
      f"{(global_stress_record[-1]-global_stress_record[0])/(global_strain_record[-1]-global_strain_record[0])/1e9:.2f} GPa ")
#%% Section-7: Plotting
## 7.1 Plotting deformed and undeformed mesh
Ux, Uy = U_final[0::2], U_final[1::2] # extracting x, and y components of U seprately
U_cord = np.array([Ux,Uy]).T
scale = config[6] # the amount of deofrmation is quite low, so to visualize the deformed mesh, we scale up the displacements
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot()
for ii in range(nel):
    elm = elements.loc[ii]
    node_ids = [elm[0], elm[1], elm[2], elm[3]]
    elm_cord = np.vstack([nodecords.loc[nid] for nid in node_ids] + [nodecords.loc[node_ids[0]]])
    dis_cord = np.vstack([U_cord[nid] for nid in node_ids] + [U_cord[node_ids[0]]])
    node2 =  nodecords.loc[elm[1]]
    #-------------- plots with lables
    undeformed_label = "Undeformed Mesh" if ii == 0 else "_nolegend_"
    deformed_label = "Deformed Mesh" if ii == 0 else "_nolegend_"
    #--------------
    ax.plot(elm_cord[:,0],elm_cord[:,1], color = "k", linestyle = "-", linewidth = 0.75, label= undeformed_label)
    ## add 
    def_cord = elm_cord+(dis_cord*scale)
    ax.plot(def_cord[:,0],def_cord[:,1], color = "r", linestyle = "--", linewidth = 0.75, label=deformed_label)
ax.set_xlim(-0.25,1.25)
ax.set_ylim(-0.1,2.5)
ax.legend(loc = "upper right", frameon = False)
ax.set_aspect("equal", adjustable="box")  
plt.savefig("Deformed_vs_undeformed_mesh.png")   
plt.close()

## 7.2 Plotting load vs displacement plot
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot()
ax.set_box_aspect(1)
ax.plot(disp_hist, force_hist, marker = "H", linestyle= "-", color = "tab:blue")
ax.set_xlabel(r"Displacement ($m$)")
ax.set_ylabel(r"Force ($N$)")
plt.savefig("load_vs_displacement.png")
plt.close()

## 7.3 Stress-Strain plot for the selected node
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot()
ax.set_box_aspect(1)
ax.plot(node_eyy_hist, np.array(node_syy_hist)/1e6, marker = "H", linestyle= "-", color = "b")
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel(r"$\sigma$ ($MPa$)")
plt.savefig("Stress_vs_Strain_plot_at_node.png")
plt.close()

        
        
        

  








    
    
    
    
    
    
        
    

