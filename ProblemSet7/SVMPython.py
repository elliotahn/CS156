# SVMPython.py
# This module takes in a pandas dataframe with data and outputs and
# returns a weight corresponding to the SVM approach. This module
# assumes that the data is separable. This module also assumes that the
# given dataframe has first column called "__const__".

# IMPORTED MODULES

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from cvxopt import matrix
from cvxopt.solvers import qp

# CONVERT DATAFRAME TO A MATRIX FORM

def getQuadMat(df, res):
    # CREATE THE SHAPE OF THE RETURN MATRIX INITIALIZED TO RANDOM
    ndat = df.shape[0]
    rmat = np.empty((ndat, ndat))

    droplst = ['__const__', res]
    XX = df.drop(droplst, axis = 1).values
    YY = np.array(df[[res]].values.reshape(1, ndat))

    for i in range(ndat):
        for j in range(ndat):
            rmat[i][j] = YY[0][i] * YY[0][j] * np.dot(XX[i], XX[j])

    return matrix(rmat), matrix(YY), XX

# MAIN FUNCTION TO BE CALLED
# Input: df - pandas dataframe
#        res - output string description
# Output : w - weight of the separable data according to SVM
# -----------------------------------------------------------
# The variables used are set to minimize the expression below with respect
# to x:
# (1/2) xT P x + qT x subject to G x <= h, A x = b

def SVMQuadProg(df, res):
    thresh = 1e-8
    ndat = df.shape[0]
    P, A, XX = getQuadMat(df, res)
    q = matrix(-1., (ndat, 1))
    b = matrix(0.)
    h = matrix(0., (ndat, 1))
    G = matrix(0.0, (ndat, ndat))
    G[::ndat + 1] = -1.

    alpha = qp(P, q, G, h, A, b)['x']

    w = np.zeros(df.shape[1] - 2)
    j = 0
    for i in range(ndat):
        if alpha[i] > thresh:
            w += alpha[i] * A[i] * XX[i]
            j = i

    w0 = 1. / A[j] - np.dot(w, XX[j])

    return np.insert(w, 0, w0)

# THIS IS AN ALTERNATIVE VERSION THAT UTILIZES SCIPY INSTEAD OF CVXOPT
# MODULES. THE SAME ASSUMPTIONS APPLY REGARDING THE FORM OF THE INPUTS

def getQuadMat2(df, res):
    # CREATE THE SHAPE OF THE RETURN MATRIX INITIALIZED TO RANDOM
    ndat = df.shape[0]
    rmat = np.empty((ndat, ndat))

    droplst = ['__const__', res]
    XX = df.drop(droplst, axis = 1).values
    YY = np.array(df[[res]].values.reshape(1, ndat))

    for i in range(ndat):
        for j in range(ndat):
            rmat[i][j] = YY[0][i] * YY[0][j] * np.dot(XX[i], XX[j])

    return rmat, YY, XX

def objective(alpha, P, YY):
    npalpha = np.array(alpha)
    return 0.5 * np.matmul(npalpha, np.matmul(P, npalpha)) - np.sum(npalpha)

def SVMQuadProg2(df, res):
    thresh = 1e-8
    ndat = df.shape[0]
    P, YY, XX = getQuadMat2(df, res)
    alpha0 = (1.,) * ndat
    
    obj = lambda x : 0.5 * np.matmul(x, np.matmul(P, x)) - np.sum(x)
    cnstrt = lambda x : np.dot(YY[0], np.array(x))
    bnd = (0., float('inf'))
    bnds = (bnd,) * ndat
    con = [{'type': 'eq', 'fun': cnstrt}]
    alpha = minimize(obj, alpha0, method = 'SLSQP', bounds = bnds,
            constraints = con)['x']
    
    w = np.zeros(df.shape[1] - 2)
    j = 0
    SVcount = 0
    for i in range(ndat):
        if alpha[i] > thresh:
            w += alpha[i] * YY[0][i] * XX[i]
            j = i
            SVcount += 1

    w0 = 1. / YY[0][j] - np.dot(w, XX[j])

    return np.insert(w, 0, w0), SVcount






