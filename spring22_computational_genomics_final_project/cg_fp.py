"""
Nick McCloskey biol 5509 course project

simulate evolution of metabolic pathway over a tree

based on Orlenko, et al. 2016
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4938953/

initial population is homogeneous
forward time simulation, discrete generations
model expressed as system of diffeqs
solve set of diffeqs for each ind each gen to calculate flux and fitness
each ind rep'd as instance of described model
    mutations that may have fitness effects
    mutational effects for everything except enzyme concentration come from normal distribution centered at -1%
    contrain by Haldane's relationship
    kcatr = (kcat * kmr) / (keq * km)
selection scheme: selection on steady state flux alone
an ind's fitness:
    F1 = 1 / (1 + -exp(flux-650)^0.07)
    flux lower than 650 is not viable, i.e. fitness = 0
    0 fitness = 0 chance of appearing in next generation
sample inds based on fitness, weighted sampling w replacement

split population based on newick string - all populations same size
    for this project, the words node and branch are fungible enough
        nodes are represented by newick (sub)strings
        simulations are run along the branch from the parent node to the child node
        so node n represents the branch from n's parent to n
    root node is populated with p homogeneous individuals
    (the drosophila tree used here is not rooted, so the trifurcation at the beginning is treated as the root for this project)
    each branching population starts with p individuals selected
        (weighted sampling with replacement) from previous branch
    branch lengths proportional to number of generations of separation
        maximum branch length (mbl) calculated from input newick string
        generations of simulation for a given branch:
            g_per_brach = g/mbl * branch_length
            rounded to nearest integer

Data:
empty dictionary passed into simulate function to fill with data
data collected:
    key = node number representing node/branch
    value = list whose first three members have themselves g_per_branch # of members
        [0] = list of average fluxes for that branch
        [1] = list of average fitnesses for that branch
        [2] = dictionary of average parameter values for that branch
        [3] = small dictionary containing average flux, fitness, and parameter values for that branch

"""
import os
import re
import copy
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class tree():
    def __init__(self,ns=''):
        self.ns = ns # newick string
        self.external_nodes = []
        self.internal_nodes = []
        self.root_node = None
        self.all_nodes = []
        self.BL = False
    
    class node():
        def __init__(self,subns=None):
            self.ancs = [] # list of all ancestors
            self.scions = [] # list of all descendants
            self.parent = None # immediate ancestor
            self.children = [] # immediate descendants
            self.sis = [] # sister node(s)
            self.subns = subns # newick substring
            self.type = None # external, internal, or root
            self.dbl = 0 # down branch length
            self.no = None # node number - for refering to its data
            self.op = None # original population
            self.bp = None # branching population

    def node_finder(self):
        
        # determine if newick string contains branch lengths
        if ':' in self.ns:
            self.BL = True

        # extract external node names
        for string in self.ns.split(','):
            substring = string.replace('(','').replace(')','').replace(',','').replace(';','')
            # remove branch lengths from anc nodes
            if substring.count(':') > 1:
                m = re.match(r'.+?\:.+?\:',substring)
                substring = m.group(0)[:-1]
            n = self.node(subns=substring)
            if self.BL:
                n.dbl = float(substring[substring.index(':')+1:])
            n.type = 'external'
            self.external_nodes.append(n)

        # internal nodes have matching parentheses
        oc_starts = [m.span()[0] for m in re.finditer(r'\(',self.ns)]
        # fop = index of first open parenthesis
        for fop in oc_starts:
            opc = 1 # open parenthesis count
            cpc = 0 # closed parenthesis count
            lcp = fop + 1 # counter for index of last closed parenthesis
            for c in self.ns[fop+1:]:
                if c == '(':
                    opc += 1
                elif c == ')':
                    cpc += 1
                lcp += 1
                if opc == cpc:
                    lcp += 1
                    break
            substring = self.ns[fop:lcp]
            if substring[-1] == ',' or substring[-1] == ';' or substring[-1] == ':':
                substring = substring[:-1]
            if substring.count('(') != substring.count(')'):
                substring = substring[:-1]
            n = self.node(subns=substring)
            n.type = 'internal'
            self.internal_nodes.append(n)

        # identify ancestors
        def sort_fun(node):
            return len(node.subns)

        self.all_nodes = self.external_nodes + self.internal_nodes
        for n1 in self.all_nodes:
            for n2 in self.all_nodes:
                if n1.subns in n2.subns and n1.subns != n2.subns:
                    n1.ancs.append(n2)
                    n2.scions.append(n1)
        for n in self.all_nodes:
            # from shortest to longest, moving down the tree
            n.ancs.sort(key=sort_fun)
            # from longest to shortest, moving up the tree
            n.scions.sort(key=sort_fun,reverse=True)
        
        # assign down branch lengths to internal nodes
        for intnode in self.internal_nodes:
            try:
                string_start = self.ns.index(intnode.subns)
                string_stop = string_start + len(intnode.subns) + 1
                m = re.match(r'.+?(\,|\))',self.ns[string_stop:])
                intnode.dbl = float(m.group()[:-1])
            except:
                pass

        # identify root node (1) and number nodes
        self.all_nodes.sort(key=sort_fun,reverse=True)
        self.root_node = self.all_nodes[0]
        self.root_node.type = 'root'
        for i,node in enumerate(self.all_nodes):
            node.no = i

        # identify sisters, parents, and children
        for n1 in self.all_nodes:
            sisters = []
            for n2 in self.all_nodes:
                if n1.ancs == n2.ancs and n1 != n2:
                    sisters.append(n2)
                    if n1 not in sisters:
                        sisters.append(n1)
            for sist in sisters:
                subsist = sisters.copy()
                subsist.remove(sist)
                sist.sis = subsist
                sist.parent = sist.ancs[0]
                if sist not in sist.ancs[0].children:
                    sist.ancs[0].children.append(sist)
            n1.children.sort(key=sort_fun)

    # simulation function
    def simulate(self,g,p,m,keqd,spd,dd):
        
        print('total branches: ' + str(len(self.all_nodes)))

        # system of differential equations
        def flux(y0,t,ipd):
            # constant supply of substrate A
            a = 1
            # initial states of other substrates = 0
            b,c,d,e,f = y0
            # parameters come from ind's param dict
            ea,kc1,kc1r,km1,km1r,ki1 = list(ipd['A'].values())
            eb,kc2,kc2r,km2,km2r = list(ipd['B'].values())
            ec,kc3,kc3r,km3,km3r = list(ipd['C'].values())
            ed,kc4,kc4r,km4,km4r = list(ipd['D'].values())
            ee,kc5,kc5r,km5,km5r = list(ipd['E'].values())
            # Michaelis-Menten kinetics
            dbdt = ((kc1*a*ea)/(a+km1*(1+b/km1r+b/ki1)) - (kc1r*b*ea)/(b+km1r*(1+a/km1+b/ki1))) - (((kc2*b*eb)/(b+km2*(1+c/km2r))) - (kc2r*c*eb)/(c+km2r*(1+b/km2)))
            dcdt = ((kc2*b*eb)/(b+km2*(1+c/km2r)) - (kc2r*c*eb)/(c+km2r*(1+b/km2))) - (((kc3*c*ec)/(c+km3*(1+d/km3r))) - (kc3r*d*ec)/(d+km3r*(1+c/km3)))
            dddt = ((kc3*c*ec)/(c+km3*(1+d/km3r)) - (kc3r*d*ec)/(d+km3r*(1+c/km3))) - (((kc4*d*ed)/(d+km4*(1+e/km4r))) - (kc4r*e*ed)/(e+km4r*(1+d/km4)))
            dedt = ((kc4*d*ed)/(d+km4*(1+e/km4r)) - (kc4r*e*ed)/(e+km4r*(1+d/km4))) - (((kc5*e*ee)/(e+km5*(1+f/km5r))) - (kc5r*f*ee)/(f+km5r*(1+e/km5)))
            dfdt = ((kc5*e*ee)/(e+km5*(1+f/km5r)) - (kc5r*f*ee)/(f+km5r*(1+e/km5))) - 0.1*f
            # flux is asymptotic values of dfdt
            return [dbdt,dcdt,dddt,dedt,dfdt]

        # create ind class for individuals in population
        class ind():
            def __init__(self):
                # dict of dicts of enzyme parameters
                # to get a parameter for an enzyme: ind.params[enzyme][parameter]
                self.params = None
                # an individual's fitness
                self.fitness = 0
            
            def copy_ind(self):
                # function to replicate an individual with deepcopy
                new_ind = ind()
                new_ind.params = copy.deepcopy(self.params)
                new_ind.fitness = self.fitness
                return new_ind
            
            def calc_flux_and_fitness(self):
                # initial (non-a) susbtrate concentrations
                y0 = [0.0] * 5
                # time points usually sufficient to reach asymptote (steady state flux)
                tp,tps = 70,700
                t = np.linspace(0,tp,tps)
                # solve system of differential equations for flux
                x = odeint(flux,y0,t,args=(self.params,))
                # extract flux - very last time point
                fx = x[-1][-1]
                # average of last 100 time points
                afx = sum(x[-100:,-1]) / 100
                # function to rerun flux calculation with adjusted t
                def reflux(tp,tps):
                    t = np.linspace(0,tp,tps)
                    x = odeint(flux,y0,t,args=(self.params,))
                    fx = x[-1][-1]
                    afx = sum(x[-100:,-1]) / 100
                    return fx,afx
                # check for asymptote - afx should differ from fx by < 0.1%
                while (np.abs(fx-afx)/fx)*100 > 0.1:
                    tp += 10
                    tps += 100
                    fx,afx = reflux(tp,tps)
                # if flux is too low, fitness is 0 (this prevents nan values for fitness)
                if fx < 650:
                    ft = 0
                else:   
                    # calculate fitness
                    ft = 1 / (1 + np.exp(-(fx-650)**0.07))
                self.fitness = ft
                return fx,ft
            
            def mutate(self):
                pard = copy.deepcopy(self.params)
                for pathway in pard:
                    kmutation = False # prompts recalculation of kcatr
                    pl = ['ec','kcat','km','kmr']
                    # only pathway A has inhibition constant
                    if pathway == 'A':
                        pl.append('ki')
                    # limited list of mutable params
                    for param in pl:
                        if random.uniform(0,1) <= m:
                            if param == 'ec':
                                u = -0.01 * np.exp(0.025*pard[pathway][param])
                                me = np.random.normal(u) * 0.01
                            else:
                                kmutation = True
                                # mutational scheme constrained by Haldane's relationship
                                me = np.random.normal(loc=-0.01,scale=0.01)
                            pard[pathway][param] *= me + 1
                    # kcatr calculated from mutated parameters
                    if kmutation:
                        # kcatr = (kcat * kmr) / (keq * km)
                        pard[pathway]['kcatr'] = (pard[pathway]['kcat'] * pard[pathway]['kmr']) / (keqd[pathway] * pard[pathway]['km'])
                        kmutation = False
                self.params = copy.deepcopy(pard)
        
        # create starting population of p homogeneous inds
        spop = []
        for j in range(p):
            i = ind()
            i.params = copy.deepcopy(spd)
            spop.append(i)
            # starting population is root node's original population
            self.root_node.op = [i.copy_ind() for i in spop]
        
        def population(self,node,g,p):
            # make pool from original population
            pool = [i.copy_ind() for i in node.op]
            # average fluxes, fitnesses, and params for the branch
            b_avg_fx = 0
            b_avg_ft = 0
            nd = {'A':{par:0 for par in ['ec','kcat','kcatr','km','kmr','ki']}}
            nd1 = {L:{par:0 for par in ['ec','kcat','kcatr','km','kmr']} for L in 'BCDE'}
            nd.update(nd1)
            b_avg_prs = copy.deepcopy(nd)
            # assemble into dictionary of branch averages
            nb_avgs = {'b_avg_fx':b_avg_fx,'b_avg_ft':b_avg_ft,'b_avg_prs':b_avg_prs}
            b_avgs = copy.deepcopy(nb_avgs)
            if g == 0: # as it is for root node
                for ind in pool: # calculate average flux, fitness, and parameters
                    fx,ft = ind.calc_flux_and_fitness()
                    b_avg_fx += (fx / p)
                    b_avg_ft += (ft / p)
                    if node.type != 'root':
                        for path in b_avg_prs:
                            for par in b_avg_prs[path]:
                                b_avg_prs[path][par] += ind.params[path][par] / p
                b_avgs['b_avg_fx'] = b_avg_fx
                b_avgs['b_avg_ft'] = b_avg_ft
                if node.type == 'root':
                    b_avgs['b_avg_prs'] = copy.deepcopy(spd)
                else:
                    b_avgs['b_avg_prs'] = copy.deepcopy(b_avg_prs)
                # first three list elements are empty because no generations transpired
                node.bp = [i.copy_ind() for i in pool]
                print('finished branch ' + str(node.no))
                return [[], [], [], b_avgs]
            else:
                # average fluxes, fitnesses, and params for each generation
                g_avg_fxs = [0] * g
                g_avg_fts = [0] * g
                nd = {'A':{par:[0]*g for par in ['ec','kcat','kcatr','km','kmr','ki']}}
                nd1 = {L:{par:[0]*g for par in ['ec','kcat','kcatr','km','kmr']} for L in 'BCDE'}
                nd.update(nd1)
                g_avg_prs = copy.deepcopy(nd)
                for i in range(g):
                    # mutate, and calculate flux and fitness for each ind in population
                    for ind in pool:
                        ind.mutate()
                        # individual data
                        i_fx,i_ft = ind.calc_flux_and_fitness()
                        # generational data
                        g_avg_fxs[i] += i_fx / p
                        g_avg_fts[i] += i_ft / p
                        for path in ind.params:
                            for par in ind.params[path]:
                                g_avg_prs[path][par][i] += ind.params[path][par] / p
                    # weighted (by fitness) sampling with replacement to make next generation
                    new_pop = random.choices(pool,weights=[i.fitness for i in pool],k=p)
                    pool = [i.copy_ind() for i in new_pop]
                # branch data
                b_avg_fx = sum(g_avg_fxs) / g
                b_avg_ft = sum(g_avg_fts) / g
                for path in g_avg_prs:
                    for par in g_avg_prs[path]:
                        b_avg_prs[path][par] = sum(g_avg_prs[path][par]) / g
                b_avgs['b_avg_fx'] = b_avg_fx
                b_avgs['b_avg_ft'] = b_avg_ft
                b_avgs['b_avg_prs'] = b_avg_prs
                # branching population
                node.bp = [i.copy_ind() for i in new_pop]
                print('finished branch ' + str(node.no))
                return [g_avg_fxs, g_avg_fts, g_avg_prs, b_avgs]

        # find ratio to convert branch length or height into generations per branch
        def find_gr(self,g):
            info = []
            for node in self.all_nodes:
                if self.BL:
                    # find max branch length
                    info.append(sum([n.dbl for n in [node] + node.ancs]))
                else:
                    # find max height
                    info.append(len(node.ancs))
            return g / max(info)

        gr = find_gr(self,g)

        # recursive function to run simulation over tree
        def branch(self,node,dd,p,gr):
            # determine how many generations to run given a branch length
            if self.BL:
                branch_g = round(gr*node.dbl)
            else:
                branch_g = round(gr*len(node.ancs))
            dd[node.no] = population(self,node,branch_g,p)
            if node.children != []:
                for child in node.children:
                    # original population of child node is branching population of parent node
                    child.op = [i.copy_ind() for i in node.bp]
                    branch(self,child,dd,p,gr)

        branch(self,self.root_node,dd,p,gr)

        return dd

# dictionary of keq values
keqd = {'A':300,'B':10,'C':0.75,'D':0.5,'E':2000}
# dictionary of starting enzyme parameters
spd = {'A':{'ec':10,'kcat':300,'kcatr':3000,'km':0.01,'kmr':30,'ki':40},
       'B':{'ec':10,'kcat':100,'kcatr':3000,'km':0.1,'kmr':30},
       'C':{'ec':10,'kcat':75,'kcatr':3000,'km':1,'kmr':30},
       'D':{'ec':10,'kcat':75,'kcatr':3000,'km':1.5,'kmr':30},
       'E':{'ec':10,'kcat':200,'kcatr':3000,'km':0.001,'kmr':30}}
g = 100 # generations
p = 100 # population size
m = 0.003 # mutation rate
dd = {} # empty dict to fill with data

# read newick string
nsf = 'drosophilatree.nwk'
with open(nsf) as file: ns = file.read().rstrip('\n')
# create tree and node objects
T = tree(ns)
T.node_finder()

# run simluation
start_time = time.time()
datetime_string = str(datetime.now()).replace(' ','_').replace(':','.')[:-10]
dd = T.simulate(g,p,m,keqd,spd,dd)
stop_time = time.time()
runtime = stop_time - start_time
print('Runtime: ' + str(runtime) + ' seconds')
if runtime / 60 > 1:
    runtime /= 60
    if runtime / 60 > 1:
        runtime /= 60
        runtime_string = str(round(runtime)) + 'hr'
    else:
        runtime_string = str(round(runtime)) + 'min'
else:
    runtime_string = str(round(runtime)) + 'sec'

"""
data and graphs, by branch and line

branch: of a non-root node, the simulation from parent node to itself
line: of an external node, the simulation from the root node through all intervening internal nodes to itself

for parameters kcat and km,
range of starting values between pathways is too varied for graphing by line or branch
so five graphs are generated, one for each pathway

for parameters ec, kcatr, kmr,
five pathway graphs as well as graphs for each line and branch are generated

datatables for final branch/line values and percent differences generated
heatmaps representing the latter generated

output folder
name: simulation_data_g_p_datetime_runtime
contents: data dictionary, generations for branch and line, 
cumulative generations for branch, average fitness and flux for branch and line 
over generations, datatables, heatmaps
subfolders branch_ and line_param_charts
    one graph of ki for path A by branch/line
    subsubfolders: ec, kcat, kcatr, km, kmr containing averages over generations
"""

# branch count
bc = len(T.all_nodes)
# branch x-axis labels
x_labels = ['root'] + [str(node.no) for node in T.all_nodes[1:]]

# directory
path = os.getcwd() + '\\simulation_data'
if not os.path.isdir(path):
    os.mkdir(path)

# save dictionary
with open('simulation_data/data_dictionary.txt','w') as file: file.write(str(dd))

# plot generations per branch
gpb = [len(dd[i][0]) for i in range(bc)]
plt.bar(x_labels,gpb)
plt.rcParams["figure.figsize"] = (14,13)
plt.rcParams["font.size"] = "20"
plt.title('Generations per Branch',fontsize=20)
plt.ylabel('No. generations',fontsize=15)
plt.xlabel('Node',fontsize=15)
plt.savefig('simulation_data/generations_per_branch.jpeg')
plt.clf()

# plot total generations per branch
tgpb = gpb.copy()
for i in range(bc):
    for anc in T.all_nodes[i].ancs:
        an = anc.no
        tgpb[i] += len(dd[an][0])
plt.bar(x_labels,tgpb)
plt.rcParams["figure.figsize"] = (14,13)
plt.rcParams["font.size"] = "20"
plt.title('Total Generations per Branch',fontsize=20)
plt.ylabel('No. generations',fontsize=15)
plt.xlabel('Node',fontsize=15)
plt.savefig('simulation_data/total_generations_per_branch.jpeg')
plt.clf()

# plot average flux per branch
b_fxs = [dd[i][3]['b_avg_fx'] for i in range(bc)]
plt.bar(x_labels,b_fxs)
plt.rcParams["figure.figsize"] = (14,13)
plt.rcParams["font.size"] = "13"
ax = plt.gca()
fxmin, fxmax = min(b_fxs), max(b_fxs)
ybuff = (fxmax - fxmin) / 2
ax.set_ylim(fxmin-ybuff,fxmax+ybuff)
plt.title('Average Flux by Branch',fontsize=22)
plt.ylabel('Enzyme Flux (mmol/l/s)',fontsize=18)
plt.xlabel('Branch',fontsize=18)
plt.savefig('simulation_data/average_flux_per_branch.jpeg')
ax.clear()
plt.clf()
# plot average flux per generation per branch
for b in range(1,bc):
    plt.plot(range(1,gpb[b]+1),dd[b][0])
    plt.rcParams["figure.figsize"] = (14,13)
plt.title('Average Flux per Branch')
plt.ylabel('Flux (mmol/l/s)')
plt.xlabel('generations')
plt.legend(range(1,bc))
plt.savefig('simulation_data/flux_per_generation.jpeg')
plt.clf()

# plot average fitness per branch
b_fts = [dd[i][3]['b_avg_ft'] for i in range(bc)]
plt.bar(x_labels,b_fts)
plt.rcParams["figure.figsize"] = (14,13)
ax = plt.gca()
ftmin, ftmax = min(b_fts), max(b_fts)
ybuff = (ftmax - ftmin) / 2
ax.set_ylim(ftmin-ybuff,ftmax+ybuff)
plt.title('Average Fitness by Branch')
plt.ylabel('Fitness')
plt.xlabel('Branch')
plt.savefig('simulation_data/average_fitness_per_branch.jpeg')
ax.clear()
plt.clf()
# plot average fitness per generation per branch
for b in range(1,bc):
    plt.plot(range(1,gpb[b]+1),dd[b][1])
    plt.rcParams["figure.figsize"] = (14,13)
plt.title('Average Fitness per Branch')
plt.ylabel('fitness')
plt.xlabel('generations')
plt.legend(range(1,bc))
plt.savefig('simulation_data/fitness_per_generation.jpeg')
plt.clf()

# plot average parameter values for each branch
path = os.getcwd() + '\\simulation_data\\branch_param_charts'
if not os.path.isdir(path):
    os.mkdir(path)

# pathways for legends
paths = [L for L in 'ABCDE']

# enzyme concentrations: 1 graph per branch, 5 paths per graph
path = os.getcwd() + '\\simulation_data\\branch_param_charts\\ec'
if not os.path.isdir(path):
    os.mkdir(path)
for b in range(1,bc):
    # enzyme concentrations for each pathway
    for x in [dd[b][2][L]['ec'] for L in 'ABCDE']:
        plt.plot(range(1,gpb[b]+1),x)
    plt.title('Average Enzyme Concentration along Branch ' + str(b))
    plt.ylabel('[Enzyme] (mmol/l)')
    plt.xlabel('generations')
    plt.legend(paths)
    plt.savefig('simulation_data/branch_param_charts/ec/ec_{}.jpeg'.format(str(b)))
    plt.clf()
# enzyme concentrations: 1 graph per path, b branches per graph
for P in paths:
    for b in range(1,bc):    
        plt.plot(range(gpb[b]),dd[b][2][P]['ec'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average Enzyme Concentration per Branch for Path %s' % P)
    plt.ylabel('[Enzyme] (mmol/l)')
    plt.xlabel('generations')
    plt.legend(range(1,bc))
    plt.savefig('simulation_data/branch_param_charts/ec/ec_%s.jpeg' % P)
    plt.clf()

# kcat: 1 graph per path, b branches per graph
path = os.getcwd() + '\\simulation_data\\branch_param_charts\\kcat'
if not os.path.isdir(path):
    os.mkdir(path)
for P in paths:
    for b in range(1,bc):    
        plt.plot(range(1,gpb[b]+1),dd[b][2][P]['kcat'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average kcat per Branch for Path %s' % P)
    plt.ylabel('kcat (mmol/l/s)')
    plt.xlabel('generations')
    plt.legend(range(1,bc))
    plt.savefig('simulation_data/branch_param_charts/kcat/kcat_%s.jpeg' % P)
    plt.clf()

# kcatr: 1 graph per branch, 5 paths per graph
path = os.getcwd() + '\\simulation_data\\branch_param_charts\\kcatr'
if not os.path.isdir(path):
    os.mkdir(path)
for b in range(1,bc):
    for x in [dd[b][2][L]['kcatr'] for L in 'ABCDE']:
        plt.plot(range(1,gpb[b]+1),x)
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average kcatr along Branch ' + str(b))
    plt.ylabel('kcatr (mmol/l/s)')
    plt.xlabel('generations')
    plt.legend(paths)
    plt.savefig('simulation_data/branch_param_charts/kcatr/kcatr_{}.jpeg'.format(str(b)))
    plt.clf()
# kcatr: 1 graph per path, b branches per graph
for P in paths:
    for b in range(1,bc):
        plt.plot(range(1,gpb[b]+1),dd[b][2][P]['kcatr'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average kcatr per Branch for Path %s' % P)
    plt.ylabel('kcatr (mmol/l/s)')
    plt.xlabel('generations')
    plt.legend(range(1,bc))
    plt.savefig('simulation_data/branch_param_charts/kcatr/kcatr_%s.jpeg' % P)
    plt.clf()

# km: 1 graph per path, b branches per graph
path = os.getcwd() + '\\simulation_data\\branch_param_charts\\km'
if not os.path.isdir(path):
    os.mkdir(path)
for P in paths:
    for b in range(1,bc):    
        plt.plot(range(1,gpb[b]+1),dd[b][2][P]['km'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average km per Branch for Path %s' % P)
    plt.ylabel('kcat (mmol/l)')
    plt.xlabel('generations')
    plt.legend(range(1,bc))
    plt.savefig('simulation_data/branch_param_charts/km/km_%s.jpeg' % P)
    plt.clf()

# kmr: 1 graph per branch, 5 paths per graph
path = os.getcwd() + '\\simulation_data\\branch_param_charts\\kmr'
if not os.path.isdir(path):
    os.mkdir(path)
for b in range(1,bc):
    for x in [dd[b][2][L]['kmr'] for L in 'ABCDE']:
        plt.plot(range(1,gpb[b]+1),x)
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average kmr along Branch ' + str(b))
    plt.ylabel('kmr (mmol/l)')
    plt.xlabel('generations')
    plt.legend(paths)
    plt.savefig('simulation_data/branch_param_charts/kmr/kmr_{}.jpeg'.format(str(b)))
    plt.clf()
# kmr: 1 graph per path, b branches per graph
for P in paths:
    for b in range(1,bc):    
        plt.plot(range(1,gpb[b]+1),dd[b][2][P]['kmr'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average kmr per Branch for Path %s' % P)
    plt.ylabel('kmr (mmol/l)')
    plt.xlabel('generations')
    plt.legend(range(1,bc))
    plt.savefig('simulation_data/branch_param_charts/kmr/kmr_%s.jpeg' % P)
    plt.clf()

# ki: 1 graph, all branches
for b in range(1,bc):
    plt.plot(range(1,gpb[b]+1),dd[b][2]['A']['ki'])
    plt.rcParams["figure.figsize"] = (14,13)
plt.title('Average ki per Branch, Path A')
plt.ylabel('ki (mmol/l)')
plt.xlabel('generations')
plt.legend(range(1,bc))
plt.savefig('simulation_data/branch_param_charts/ki_A.jpeg')
plt.clf()

# output datatable of average parameters per branch
allprs = ['ec','kcat','kcatr','km','kmr','ki']
mostprs = ['ec','kcat','kcatr','km','kmr']
# extract average parameter values from each node
arows = [[dd[i][3]['b_avg_prs']['A'][par] for par in allprs] for i in range(bc)]
orows = [[dd[i][3]['b_avg_prs'][path][par] for path in 'BCDE' for par in mostprs] for i in range(bc)]
for arow,orow in zip(arows,orows):
    for a in arow[::-1]:
        orow.insert(0,a)
# extract substrings, branch lengths, generations per branch
nss = ['root (input Newick string)'] + [n.subns for n in T.all_nodes[1:]]
bls = [0] + [n.dbl for n in T.all_nodes[1:]]
gpbs = [0] + [len(dd[i][0]) for i in range(bc)]
# insert above info plus average flux and fitness
for ns,bl,gbp,afx,aft,row in zip(nss,bls,gpb,b_fxs,b_fts,orows):
    for d in [ns,bl,gbp,afx,aft][::-1]:
        row.insert(0,d)
columns = ['Newick substring','branch length','generations','average flux','average fitness'] + ['%s-%s' % (path,par) for path,par in zip('A'*6,['ec','kcat','kcatr','km','kmr','ki'])] + ['%s-%s' % (path,par) for path,par in zip('B'*5+'C'*5+'D'*5+'E'*5,['ec','kcat','kcatr','km','kmr']*5)]
df = pd.DataFrame(orows,columns=columns,index=(range(bc)))
df.index.name = 'branch'
df.to_excel('simulation_data/branch_data.xlsx')

# heatmap - % differences between start and end values
arows = [[spd['A'][par] for par in allprs] for i in range(bc)]
orows = [[spd[path][par] for path in 'BCDE' for par in mostprs] for i in range(bc)]
for arow,orow in zip(arows,orows):
    for a in arow[::-1]:
        orow.insert(0,a)
start_fx, start_ft = dd[0][3]['b_avg_fx'], dd[0][3]['b_avg_ft']
for row in orows:
    row.insert(0,start_ft)
    row.insert(0,start_fx)
startvals = np.asarray(orows)
endvals = np.asarray(df)[:,3:]
valdiffs = np.subtract(endvals,startvals)
valquotients = np.divide(valdiffs,startvals)
pdiffs = valquotients * 100
# save file of percent differences
df1 = pd.DataFrame(pdiffs,columns=columns[3:],index=(range(bc)))
df1.index.name = 'branch'
df1.to_excel('simulation_data/percent_differences_from_start_values.xlsx')
hmd = pdiffs.astype('float')
hmc = ['avgflux'] + ['avgfit'] + columns[5:]
fig, ax = plt.subplots(figsize=(14,14))
sns.heatmap(hmd,xticklabels=hmc,yticklabels=x_labels,linewidth=0.5,cmap='gnuplot2_r',ax=ax)
plt.title('Percent change in simulation values from start to end')
plt.ylabel('branch')
plt.xlabel('value')
plt.savefig('simulation_data/percent_diff_heatmap.jpeg')
plt.clf()
ax.clear()
fig.clear()

# heatmap - % differences between start and end values without avgfx or avgft
arows = [[spd['A'][par] for par in allprs] for i in range(bc)]
orows = [[spd[path][par] for path in 'BCDE' for par in mostprs] for i in range(bc)]
for arow,orow in zip(arows,orows):
    for a in arow[::-1]:
        orow.insert(0,a)
startvals = np.asarray(orows)
endvals = np.asarray(df)[:,5:]
valdiffs = np.subtract(endvals,startvals)
valquotients = np.divide(valdiffs,startvals)
pdiffs = valquotients * 100
hmd = pdiffs.astype('float')
hmc = columns[5:]
ax = sns.heatmap(hmd,xticklabels=hmc,yticklabels=x_labels,linewidth=0.5,cmap='gnuplot2_r')
plt.title('Percent change in simulation parameters from start to end')
plt.ylabel('branch')
plt.xlabel('value')
plt.savefig('simulation_data/percent_diff_heatmap1.jpeg')
plt.clf()
ax.clear()

# extract line data
ldd = copy.deepcopy(dd)
nldd = {}
enns = [node.no for node in T.external_nodes]
anc_nos = [[node.no for node in en.ancs if node.no != 0] for en in T.external_nodes]
for enn,ans in zip(enns,anc_nos):
    nldd[enn] = ldd[enn].copy()
    for an in ans:
        nldd[enn][0] = ldd[an][0] + nldd[enn][0]
        nldd[enn][1] = ldd[an][1] + nldd[enn][1]
        for path in nldd[enn][2]:
            for par in nldd[enn][2][path]:
                nldd[enn][2][path][par] = ldd[an][2][path][par] + nldd[enn][2][path][par]

# plot average parameter values for each line
path = os.getcwd() + '\\simulation_data\\line_param_charts'
if not os.path.isdir(path):
    os.mkdir(path)

# plot generations per line
gpl = {e:len(nldd[e][0]) for e in enns}
plt.bar(enns,list(gpl.values()))
plt.rcParams["figure.figsize"] = (14,13)
plt.rcParams["font.size"] = "20"
plt.title('Generations per Line',fontsize=20)
plt.ylabel('No. generations',fontsize=15)
plt.xlabel('Line (external node)',fontsize=15)
plt.xticks(enns)
plt.savefig('simulation_data/generations_per_line.jpeg')
plt.clf()

# plot average flux per line
b_fxs = [nldd[e][3]['b_avg_fx'] for e in enns]
plt.bar(enns,b_fxs)
plt.rcParams["figure.figsize"] = (14,13)
plt.rcParams["font.size"] = "13"
ax = plt.gca()
fxmin, fxmax = min(b_fxs), max(b_fxs)
ybuff = (fxmax - fxmin) / 2
ax.set_ylim(fxmin-ybuff,fxmax+ybuff)
plt.title('Average Flux by Line',fontsize=22)
plt.ylabel('Enzyme Flux (mmol/l/s)',fontsize=18)
plt.xlabel('Line (external node)',fontsize=18)
plt.xticks(enns)
plt.savefig('simulation_data/average_flux_by_line.jpeg')
ax.clear()
plt.clf()
# plot average flux per generation per line
for e in enns:
    plt.plot(range(1,gpl[e]+1),nldd[e][0])
    plt.rcParams["figure.figsize"] = (14,13)
plt.title('Average Flux per Line')
plt.ylabel('Flux (mmol/l/s)')
plt.xlabel('generations')
plt.legend(enns)
plt.savefig('simulation_data/flux_by_line_by_generation_.jpeg')
plt.clf()

# plot average fitness per line
b_fts = [nldd[e][3]['b_avg_ft'] for e in enns]
plt.bar(enns,b_fts)
plt.rcParams["figure.figsize"] = (14,13)
ax = plt.gca()
ftmin, ftmax = min(b_fts), max(b_fts)
ybuff = (ftmax - ftmin) / 2
ax.set_ylim(ftmin-ybuff,ftmax+ybuff)
plt.title('Average Fitness by Line')
plt.ylabel('Fitness')
plt.xlabel('Line (external node)')
plt.xticks(enns)
plt.savefig('simulation_data/average_fitness_per_line.jpeg')
ax.clear()
plt.clf()
# plot average fitness per generation per line
for enn in enns:
    plt.plot(range(1,gpl[enn]+1),nldd[enn][1])
    plt.rcParams["figure.figsize"] = (14,13)
plt.title('Average Fitness over Line Generations')
plt.ylabel('Flux (mmol/l/s)')
plt.xlabel('generations')
plt.legend(enns)
plt.savefig('simulation_data/fitness_by_line_by_generation_.jpeg')
plt.clf()

# line enzyme concentrations: 1 graph per branch, 5 paths per graph
path = os.getcwd() + '\\simulation_data\\line_param_charts\\ec'
if not os.path.isdir(path):
    os.mkdir(path)
for e in enns:
    # x-axis: generations for that line
    gs = [i for i in range(1,gpl[e]+1)]
    # enzyme concentrations for each pathway
    for x in [nldd[e][2][L]['ec'] for L in 'ABCDE']:
        plt.plot(gs,x)
    plt.title('Average Enzyme Concentration along Line ' + str(e))
    plt.ylabel('[Enzyme] (mmol/l)')
    plt.xlabel('generations')
    plt.legend(paths)
    plt.savefig('simulation_data/line_param_charts/ec/ec_{}.jpeg'.format(str(e)))
    plt.clf()
# enzyme concentrations: 1 graph per path, len(enns) lines per graph
for P in paths:
    for e in enns:    
        plt.plot(range(1,gpl[e]+1),nldd[e][2][P]['ec'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average Line Enzyme Concentration for Path %s' % P)
    plt.ylabel('[Enzyme] (mmol/l)')
    plt.xlabel('generations')
    plt.legend(enns)
    plt.savefig('simulation_data/line_param_charts/ec/ec_%s.jpeg' % P)
    plt.clf()

# kcat: 1 graph per path, len(enns) lines per graph
path = os.getcwd() + '\\simulation_data\\line_param_charts\\kcat'
if not os.path.isdir(path):
    os.mkdir(path)
for P in paths:
    for e in enns:    
        plt.plot(range(1,gpl[e]+1),nldd[e][2][P]['kcat'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average Line kcat for Path %s' % P)
    plt.ylabel('kcat (mmol/l/s)')
    plt.xlabel('generations')
    plt.legend(enns)
    plt.savefig('simulation_data/line_param_charts/kcat/kcat_%s.jpeg' % P)
    plt.clf()

# kcatr: 1 graph per line, 5 paths per graph
path = os.getcwd() + '\\simulation_data\\line_param_charts\\kcatr'
if not os.path.isdir(path):
    os.mkdir(path)
for e in enns:
    # gs = [i for i in range(gpb[b])]
    for x in [nldd[e][2][L]['kcatr'] for L in 'ABCDE']:
        plt.plot(range(1,gpl[e]+1),x)
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average kcatr along Line ' + str(b))
    plt.ylabel('kcatr (mmol/l/s)')
    plt.xlabel('generations')
    plt.legend(paths)
    plt.savefig('simulation_data/line_param_charts/kcatr/kcatr_{}.jpeg'.format(str(e)))
    plt.clf()
# kcatr: 1 graph per path, len(enns) per graph
for P in paths:
    for e in enns:
        plt.plot(range(1,gpl[e]+1),nldd[e][2][P]['kcatr'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average Line kcatr for Path %s' % P)
    plt.ylabel('kcatr (mmol/l/s)')
    plt.xlabel('generations')
    plt.legend(enns)
    plt.savefig('simulation_data/line_param_charts/kcatr/kcatr_%s.jpeg' % P)
    plt.clf()

# km: 1 graph per path, b branches per graph
path = os.getcwd() + '\\simulation_data\\line_param_charts\\km'
if not os.path.isdir(path):
    os.mkdir(path)
for P in paths:
    for e in enns:    
        plt.plot(range(1,gpl[e]+1),nldd[e][2][P]['km'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average Line km for Path %s' % P)
    plt.ylabel('kcat (mmol/l)')
    plt.xlabel('generations')
    plt.legend(enns)
    plt.savefig('simulation_data/line_param_charts/km/km_%s.jpeg' % P)
    plt.clf()

# kmr: 1 graph per line, 5 paths per graph
path = os.getcwd() + '\\simulation_data\\line_param_charts\\kmr'
if not os.path.isdir(path):
    os.mkdir(path)
for e in enns:
    # gs = [i for i in range(gpb[b])]
    for x in [nldd[e][2][L]['kmr'] for L in 'ABCDE']:
        plt.plot(range(1,gpl[e]+1),x)
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average kmr along Line ' + str(e))
    plt.ylabel('kmr (mmol/l)')
    plt.xlabel('generations')
    plt.legend(paths)
    plt.savefig('simulation_data/line_param_charts/kmr/kmr_{}.jpeg'.format(str(e)))
    plt.clf()
# kmr: 1 graph per path, len(enns) lines per graph
for P in paths:
    for e in enns:    
        plt.plot(range(1,gpl[e]+1),nldd[e][2][P]['kmr'])
        plt.rcParams["figure.figsize"] = (14,13)
    plt.title('Average Line kmr for Path %s' % P)
    plt.ylabel('kmr (mmol/l)')
    plt.xlabel('generations')
    plt.legend(enns)
    plt.savefig('simulation_data/line_param_charts/kmr/kmr_%s.jpeg' % P)
    plt.clf()

# ki: 1 graph, all branches
for e in enns:
    plt.plot(range(1,gpl[e]+1),nldd[e][2]['A']['ki'])
    plt.rcParams["figure.figsize"] = (14,13)
plt.title('Average Line ki, Path A')
plt.ylabel('ki (mmol/l)')
plt.xlabel('generations')
plt.legend(range(1,bc))
plt.savefig('simulation_data/line_param_charts/ki_A.jpeg')
plt.clf()

# output datatable of average parameters per line
allprs = ['ec','kcat','kcatr','km','kmr','ki']
mostprs = ['ec','kcat','kcatr','km','kmr']
# extract average parameter values from each line
arows = [[nldd[e][3]['b_avg_prs']['A'][par] for par in allprs] for e in enns]
orows = [[nldd[e][3]['b_avg_prs'][path][par] for path in 'BCDE' for par in mostprs] for e in enns]
for arow,orow in zip(arows,orows):
    for a in arow[::-1]:
        orow.insert(0,a)
# extract substrings, cumulative branch lengths, generations per line
lnss = [n.subns for n in T.external_nodes]
cbls = [sum([n.dbl for n in T.all_nodes[e].ancs] + [T.all_nodes[e].dbl]) for e in enns]
lgpl = list(gpl.values())
# insert above info plus average flux and fitness
for lns,cbl,gpcl,lafx,laft,row in zip(lnss,cbls,lgpl,b_fxs,b_fts,orows):
    for d in [lns,cbl,gpcl,lafx,laft][::-1]:
        row.insert(0,d)
columns = ['Newick substring','cumulative branch length','generations','average flux','average fitness'] + ['%s-%s' % (path,par) for path,par in zip('A'*6,['ec','kcat','kcatr','km','kmr','ki'])] + ['%s-%s' % (path,par) for path,par in zip('B'*5+'C'*5+'D'*5+'E'*5,['ec','kcat','kcatr','km','kmr']*5)]
df = pd.DataFrame(orows,columns=columns,index=(enns))
df.index.name = 'line'
df.to_excel('simulation_data/line_data.xlsx')

# heatmap - % differences between start and end values
arows = [[spd['A'][par] for par in allprs] for e in enns]
orows = [[spd[path][par] for path in 'BCDE' for par in mostprs] for e in enns]
for arow,orow in zip(arows,orows):
    for a in arow[::-1]:
        orow.insert(0,a)
start_fx, start_ft = dd[0][3]['b_avg_fx'], dd[0][3]['b_avg_ft']
for row in orows:
    row.insert(0,start_ft)
    row.insert(0,start_fx)
startvals = np.asarray(orows)
endvals = np.asarray(df)[:,3:]
valdiffs = np.subtract(endvals,startvals)
valquotients = np.divide(valdiffs,startvals)
pdiffs = valquotients * 100
# save file of percent differences
df1 = pd.DataFrame(pdiffs,columns=columns[3:],index=enns)
df1.index.name = 'line'
df1.to_excel('simulation_data/line_percent_differences_from_start_values.xlsx')
hmd = pdiffs.astype('float')
hmc = ['avgflux'] + ['avgfit'] + columns[5:]
ax = sns.heatmap(hmd,xticklabels=hmc,yticklabels=enns,linewidth=0.5,cmap='gnuplot2_r')
plt.title('Percent change in line simulation values from start to end')
plt.ylabel('line')
plt.xlabel('value')
plt.savefig('simulation_data/line_percent_diff_heatmap.jpeg')
plt.clf()
ax.clear()

# heatmap - % differences between start and end values without avgfx or avgft
arows = [[spd['A'][par] for par in allprs] for e in enns]
orows = [[spd[path][par] for path in 'BCDE' for par in mostprs] for e in enns]
for arow,orow in zip(arows,orows):
    for a in arow[::-1]:
        orow.insert(0,a)
startvals = np.asarray(orows)
endvals = np.asarray(df)[:,5:]
valdiffs = np.subtract(endvals,startvals)
valquotients = np.divide(valdiffs,startvals)
pdiffs = valquotients * 100
hmd = pdiffs.astype('float')
hmc = columns[5:]
ax = sns.heatmap(hmd,xticklabels=hmc,yticklabels=enns,linewidth=0.5,cmap='gnuplot2_r')
plt.title('Percent change in line simulation parameters from start to end')
plt.ylabel('line')
plt.xlabel('value')
plt.savefig('simulation_data/line_percent_diff_heatmap1.jpeg')
plt.clf()
ax.clear()

# rename directory to include input g, input p, datetime of simulation, and runtime
path = os.getcwd() + '\\simulation_data'
path1 = os.getcwd() + '\\simulation_data_g{}_p{}_{}_{}'.format(g,p,datetime_string,runtime_string)
os.rename(path,path1)
