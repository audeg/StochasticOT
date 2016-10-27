
############## ############## ############## ############## ############## 
########## Unregularized vs Regularized for various epsilon ############## 
############## ############## ############## ############## ############## 
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
import sys


# ## Random Setting

# In[2]:

def sample_rho_batch(rho_list,nsamples):
    sample = np.zeros([nsamples,D])
    for i in range(nsamples):
        rand = np.random.rand(1)
        idx = int(np.floor(nrho * rand))
        sample[i,:] = rho_list[idx].rvs()
        
    return sample

def sample_rho(rho_list):
    rand = np.random.rand(1)
    idx = int(np.floor(nrho * rand))
    sample = rho_list[idx].rvs()

    return sample


# ## Standard OT

# In[10]:

def area(idx,rho,X,v):
    n_samples = 10**3
    Xv = np.c_[X,np.sqrt(-v-np.min(-v))]
    kdTree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Xv)
    sample_rho(rho_list)    
    if n_samples == 1:
        Yv = np.hstack([Y,0])
    else:
        Yv = np.c_[Y,np.zeros(n_samples)]    
    neighbors = kdTree.kneighbors(Yv,n_neighbors=1,return_distance=False)
    area = np.sum(neighbors == idx) / n_samples    
    return area

def area_vect(rho,X,v,n_samples):
    Xv = np.c_[X,np.sqrt(-v-np.min(-v))]
    kdTree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Xv)
    Y = sample_rho(rho_list)
    if n_samples == 1:
        Yv = np.reshape(np.hstack([Y,0]),[1,-1])
    else:
        Yv = np.c_[Y,np.zeros(n_samples)]
    neighbors = kdTree.kneighbors(Yv,n_neighbors=1,return_distance=False)
    area_vect = np.zeros(n_target)
    for i in range(n_samples):
          area_vect[neighbors[i]] += 1
    area_vect = area_vect/n_samples   
    return area_vect

    




# ## Regularized OT

# In[16]:

def gradient(v_eps,epsilon):
    expv = np.zeros(n_target)
    while np.sum(expv) == 0:
        Y = sample_rho(rho_list)
        z = np.max((v_eps-np.sum((X-Y)**2,axis=1))/epsilon)
        expv = nu * np.exp((v_eps-np.sum((X-Y)**2,axis=1))/epsilon - z)
        #z = np.max(v_eps-np.sum((X-Y)**2,axis=1)/epsilon)
        #expv = nu * np.exp(v_eps-np.sum((X-Y)**2,axis=1)/epsilon - z)
        #if np.sum(expv) == 0:
        #print "simulate again"
    pi = expv/np.sum(expv)
    grad = - nu + pi
    return grad


# In[17]:

def runSGD (epsilon,nb_iter) :
    grad_type = "one sample "

    alpha = .8
    #alpha = 1./epsilon
    n_eps = len(eps_list)


    vlist = np.zeros([n_target,nb_iter])

    v_eps_bar = np.ones(n_target)
    v_eps = np.ones(n_target)
    
    t = time.time()
    
    for i in range(nb_iter) :
        vlist[:,i] = v_eps_bar
        #vlist[:,i] = epsilon * v_eps_bar
        step = alpha*(1./np.sqrt(i+1))
        grad = gradient(v_eps,epsilon)
        v_eps = v_eps - step*grad
        v_eps_bar = 1./(i+1)*(v_eps + i*v_eps_bar)
        
    tt = time.time()-t
    
    print ("epsilon = "+str(epsilon)+', time elapsed : '+str(tt))
    return vlist


def runBench(n_target, i_run, n_iter_comparaison, n_iter_SGD_opt):

    # In[15]:

    alpha = .8

    nb_iter_unreg = n_iter_SGD_opt


    v_bar = np.zeros(n_target)
    v = np.zeros(n_target)
    vlist = np.zeros([n_target,nb_iter_unreg])

    t = time.time()
    for i in range(nb_iter_unreg) :
        vlist[:,i] = v_bar
        n_samples = 1 
        step = alpha*(1./np.sqrt(i+1))
        grad = - nu + area_vect(rho,X,v,n_samples)
        v = v - step*grad
        v_bar = 1./(i+1)*(v + i*v_bar)
        
    v_opt = v_bar

    tt = time.time()-t
    
    print ('Unregularized, time elapsed : '+str(tt))


    n_it_SGD = n_iter_comparaison

    v_SGD = np.zeros([n_target,n_it_SGD,n_eps])

    for i in range(n_eps):
        epsilon = eps_list[i]
        v_SGD[:,:,i] = runSGD(epsilon,n_it_SGD)


    kmax = np.shape(v_SGD)[1]
    lmax = np.shape(vlist)[1]

    n_size_err = min(kmax,lmax)


    err_SGD = np.zeros([n_size_err,n_eps])
    err_v = np.zeros(n_size_err)

    for l in range(n_size_err):
        err_v[l] = np.linalg.norm(vlist[:,l] - np.mean(vlist[:,l]) - v_opt + np.mean(v_opt))/np.linalg.norm(v_opt - np.mean(v_opt))

    for i in range(n_eps):
        epsilon = eps_list[i]
        for k in range(n_size_err):
            err_SGD[k,i] = np.linalg.norm(v_SGD[:,k,i]- np.mean(v_SGD[:,k,i]) - v_opt + np.mean(v_opt))/np.linalg.norm(v_opt-np.mean(v_opt))


    filename_reg = "/home/marco/temp/numpy_arrays/SemiDiscretRegularization/err_SGD_unreg_"+str(i_run)+'_batch_'+str(arg)
    np.save(filename_reg,err_v)

   
    filename = "/home/marco/temp/numpy_arrays/SemiDiscretRegularization/err_SGD_all_eps_"+str(i_run)+'_batch_'+str(arg)
    np.save(filename,err_SGD)



#########   SETTING    #########

D = 3

# continuous measure

np.random.seed(3)

rho_list = []
nrho = 3
for i in range(nrho):
    mu1 = np.random.rand(D)
    sigma_tmp = np.random.rand(D,D)
    sigma1 = 0.01 *((sigma_tmp.T + sigma_tmp)+ D * np.eye(D))
    rho = multivariate_normal(mean = mu1,cov = sigma1)
    rho_list.append(rho)


    
rho_list_target = []
nrho = 3
for i in range(nrho):
    mu1 = np.random.rand(D)
    sigma_tmp = np.random.rand(D,D)
    sigma1 = 0.01 *((sigma_tmp.T + sigma_tmp)+ D * np.eye(D))
    rho = multivariate_normal(mean = mu1,cov = sigma1)
    rho_list_target.append(rho)


# discrete measure
n_target = 10 # number of diracs
X = sample_rho_batch(rho_list_target,n_target)


nu = np.random.rand(n_target)
nu = nu/np.sum(nu) # weights of diracs



# In[9]:

eps_list = [10**(-1),10**(-2),10**(-3),10**(-4)]
#eps_list = [10**(-1),10**(-2)]

n_eps = len(eps_list)




n_iter_SGD_opt = 10**7
n_iter_comparaison = 5*10**5

nruns = 5
arg = sys.argv[1]


for i_run in range(nruns):
 
    print "-------------    "+str(i_run)+"   ----------"

    X_target = sample_rho_batch(rho_list_target,n_target)

 
    runBench(n_target, i_run, n_iter_comparaison, n_iter_SGD_opt)









