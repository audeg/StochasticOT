
# coding: utf-8

# In[1]:

#from __future__ import division


import numpy as np
import time
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from random import randint
import sklearn as sk
from sklearn.neighbors import NearestNeighbors
from scipy.signal import convolve2d
import pdb
import sys



################################################################################
###########################       TOOLBOX     ##################################
################################################################################



def runSinkhorn(epsilon,C,p,q,n_it,n_target):
    K = (np.diag(p).dot(np.exp(-C/epsilon))).dot(np.diag(q))

    b = np.ones(n_target)
    err_a = np.ones(n_it)
    err_b = np.ones(n_it)
    u_list = np.ones([n_sink,n_it ])
    v_list = np.ones([n_target,n_it])


    for i in range(n_it):
        a = p/(K.dot(b))
        b = q/((np.transpose(K)).dot(a))
        u = epsilon * np.log(a)
        v = epsilon * np.log(b)
        u_list[:,i] = u
        v_list[:,i] = v
                    

   
    u = epsilon * np.log(a)
    v = epsilon * np.log(b)
    
    return [u_list,v_list]

# ## Regularized OT

# In[30]:

def gradient_SAG(v_eps,epsilon,n_target, n_source, X_source,X_target,nu,idx,p):
    expv = np.zeros(n_target)
    while np.sum(expv) == 0:
        z = np.max(v_eps-np.sum(abs(X_target-X_source[idx,:])**p,axis=1)/epsilon)
        expv = nu * np.exp(v_eps-np.sum(abs(X_target-X_source[idx,:])**p,axis=1)/epsilon - z)
        if np.sum(expv) == 0:
            print "simulate again"
    pi = expv/np.sum(expv)
    grad = - nu + pi
    return grad

def runSAG (epsilon,nb_iter,n_target,n_source,X_target,X_source,nu,alpha) :

    v_list = np.zeros([n_target,nb_iter])    

    v_eps_bar = np.ones(n_target)
    v_eps = np.ones(n_target)

    grad_vect = np.zeros([n_target,n_source])
    grad_moy = np.zeros(n_target)

 
    for i in range(nb_iter):

        if i<n_source:
            n_grad = i+1
            idx = i
        else :
            n_grad = n_source
            idx = np.random.choice(range(n_source))

        v_list[:,i] = epsilon * v_eps
        grad_idx = gradient_SAG(v_eps,epsilon,n_target,n_source,X_source,X_target,nu,idx,p)
        grad_moy = grad_moy - grad_vect[:,idx]
        grad_vect[:,idx] = grad_idx
        grad_moy = grad_moy + grad_idx

        v_eps = v_eps - alpha/float(n_grad) * grad_moy

    v_SAG = epsilon * np.array(v_eps)

    return v_list




def gradient(v_eps,epsilon,n_target,rho_list_source,X_target,nu):
    expv = np.zeros(n_target)
    while np.sum(expv) == 0:
        Y = sample_rho(rho_list_source)
        z = np.max(v_eps-np.sum((X_target-Y)**2,axis=1)/epsilon)
        expv = nu * np.exp(v_eps-np.sum((X_target-Y)**2,axis=1)/epsilon - z)
        #if np.sum(expv) == 0:
        #print "simulate again"
    pi = expv/np.sum(expv)
    grad = - nu + pi
    return grad


def runSGD (epsilon,nb_iter,n_target,rho_list_source,X_target,nu,alpha) :

    vlist = np.zeros([n_target,nb_iter])

    v_eps_bar = np.ones(n_target)
    v_eps = np.ones(n_target)
        
    for i in range(nb_iter) :
        vlist[:,i] = epsilon * v_eps_bar
        step = alpha*(1./np.sqrt(i+1))
        grad = gradient(v_eps,epsilon,n_target,rho_list_source,X_target,nu)
        v_eps = v_eps - step*grad
        v_eps_bar = 1./(i+1)*(v_eps + i*v_eps_bar)
      
    return vlist

def sample_rho_batch(rho_list,nsamples):
    sample = np.zeros([nsamples,D])
    nrho = len(rho_list)
    for i in range(nsamples):
        rand = np.random.rand(1)
        idx = int(np.floor(nrho * rand))
        sample[i,:] = rho_list[idx].rvs()
        
    return sample

def sample_rho(rho_list):
    nrho = len(rho_list)
    rand = np.random.rand(1)
    idx = int(np.floor(nrho * rand))
    sample = rho_list[idx].rvs()

    return sample



def runBench(n_target, n_source, i_run, n_iter_comparaison, first = False, v_opt_list = None , n_iter_SGD_opt = 0):

    setting_name = "Semi-discrete D = "+str(D)+" - n_source = "+str(n_source)+" - n_target = "+str(n_target)

    print setting_name
    print '************************'


    # continuous measure

        
    # if D == 2 :
    #     x, y = np.mgrid[-0.3:1.3:0.01, -0.3:1.3:0.01]
    #     N = np.size(x)
    #     grid_mat = np.empty(x.shape + (2,))
    #     grid_mat[:, :, 0] = x; grid_mat[:, :, 1] = y
    #     rho_mat = np.zeros(np.shape(x))
    #     for i in range(nrho):
    #         rho_mat += 1./nrho * rho_list[i].pdf(grid_mat)
    #     #plt.figure()
    #     #plt.contourf(x, y, rho_mat)
    #     rho_vect = np.reshape(rho_mat,N)
    #     rho_vect = rho_vect/np.sum(rho_vect)
    #     grid_vect = np.reshape(grid_mat,[N,2])
        
    # discrete measure
    # number of diracs

    nu = np.ones(n_target)
    nu = nu/np.sum(nu) # weights of diracs

    # sample of continuous measure for Sinkhorn
     # number of diracs
    X_source = sample_rho_batch(rho_list_source,n_source)  # coordinates of diracs

    mu = np.ones(n_source)
    mu = mu/np.sum(mu) # weights of diracs


    # Run SAG
    n_it_SAG = n_iter_comparaison

    
    n_alpha_SAG = 1
    v_SAG = np.zeros([n_target,n_it_SAG ,n_eps,n_alpha_SAG])

    for i in range(n_eps):
        epsilon = eps_list[i]
        alpha_list_SAG = [0.003/epsilon]
        n_alpha_SAG = len(alpha_list_SAG)
        for j in range(n_alpha_SAG):
            alpha = alpha_list_SAG[j]
            t = time.time()
            v_SAG[:,:,i,j] = runSAG(epsilon,n_it_SAG,n_target,n_source,X_target,X_source,nu,alpha)
            tt = time.time() - t
            print ("SAG, epsilon = "+str(epsilon)+', time elapsed : '+str(tt))

    #pdb.set_trace()


    # Run SGD

    n_it_SGD = n_iter_SGD_opt
    n_alpha = 1

    if first :

        v_SGD = np.zeros([n_target,n_it_SGD ,n_eps,n_alpha])

        for i in range(n_eps):
            epsilon = eps_list[i]
            alpha_list = [.5/epsilon]
            n_alpha = len(alpha_list)
            for j in range(n_alpha):
                alpha = alpha_list[j]
                t = time.time()
                v_SGD[:,:,i,j] = runSGD(epsilon,n_it_SGD,n_target,rho_list_source,X_target,nu,alpha)
                tt = time.time() - t
                print ("SGD, epsilon = "+str(epsilon)+', time elapsed : '+str(tt))
                v_opt_list = np.array(v_SGD[:,-1,:,-1])

    #pdb.set_trace()




    # Compute error for SGD and SAG

    n_size_err = min(n_it_SGD,n_it_SAG)

    err_SGD = np.zeros([n_size_err,n_eps,n_alpha])

    err_SAG = np.zeros([n_size_err,n_eps,n_alpha_SAG])

    
    
    for a in range(n_eps):
        epsilon = eps_list[a]
        v_opt = v_opt_list[:,a] - np.mean(v_opt_list[:,a])
        for j in range(n_size_err):
            for i_alpha in range(n_alpha_SAG):
                err_SAG[j,a,i_alpha] = np.linalg.norm(v_SAG[:,j,a,i_alpha] - np.mean(v_SAG[:,j,a,i_alpha]) - v_opt )/np.linalg.norm(v_opt)
        if first :
            for k in range(n_size_err):
                for j in range(n_alpha):
                    err_SGD[k,a,j] = np.linalg.norm(v_SGD[:,k,a,j] - np.mean(v_SGD[:,j,a,j]) - v_opt )/np.linalg.norm(v_opt)
        a+=1

    if first :
        filenameSGD = "/home/marco/temp/numpy_arrays/SemiDiscretvsSAG/err_SGD_SD"+'_run_'+str(i_run)+'_batch_'+str(arg)
        np.save(filenameSGD,err_SGD)

    filename = "/home/marco/temp/numpy_arrays/SemiDiscretvsSAG/err_SAG_SD_"+str(n_source)+'_run_'+str(i_run)+'_batch_'+str(arg)
    np.save(filename,err_SAG)

    


    print '************************'
    return [v_opt_list]



########################################################################
#####################       Semidiscrete    ############################
########################################################################

D = 3
p = 2

#np.random.seed(1)

rho_list_source = []
nrho = 3
for i in range(nrho):
    mu1 = np.random.rand(D)
    sigma_tmp = np.random.rand(D,D)
    sigma1 = 0.01 *((sigma_tmp.T + sigma_tmp)+ D * np.eye(D))
    rho = multivariate_normal(mean = mu1,cov = sigma1)
    rho_list_source.append(rho)

rho_list_target = []
nrho = 3
for i in range(nrho):
    mu1 = np.random.rand(D)
    sigma_tmp = np.random.rand(D,D)
    sigma1 = 0.01 *((sigma_tmp.T + sigma_tmp)+ D * np.eye(D))
    rho = multivariate_normal(mean = mu1,cov = sigma1)
    rho_list_target.append(rho)

eps_list = [10**(-2)]
n_eps = len(eps_list)

n_target = 10

n_iter_SGD_opt = 10**7
n_iter_comparaison = 5*10**5

nruns = 5

arg = sys.argv[1]

for i_run in range(nruns):
 
    print "-------------    "+str(i_run)+"   ----------"

    X_target = sample_rho_batch(rho_list_target,n_target)

    n_source0 = 10**2

    v_opt_SGD = runBench(n_target,n_source0, i_run, n_iter_comparaison, first = True, n_iter_SGD_opt = n_iter_SGD_opt)

    n_source_list = [10**3,10**4]

    for n_source in n_source_list :
        runBench(n_target,n_source, i_run, n_iter_comparaison, v_opt_list = v_opt_SGD[0],n_iter_SGD_opt = n_iter_SGD_opt)








