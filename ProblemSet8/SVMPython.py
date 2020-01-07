# SVMPython.py
# This module takes in a pandas dataframe with data and outputs and
# returns a weight corresponding to the SVM approach. This module
# assumes that the data is separable. This module also assumes that the
# given dataframe has first column called "__const__".

# IMPORTED MODULES

import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt.solvers import qp

class supportVector:
    def __init__(self, alpha = None, X = None, y = None):
        self.alpha = alpha
        self.X = X
        self.y = y

    def changeSV(self, alpha = None, X = None, y = None):
        if alpha != None:
            self.alpha = alpha
        if X != None:
            self.X = X
        if y != None:
            self.y = y

class SVMModel:
    def __init__(self, train_dat = None, target_name = None, kernel = None, 
                    C = None):
        self.train_dat = train_dat
        self.target_name = target_name
        self.kernel = kernel
        self.C = C
        self.model = None

    def train_data(self, train_dat = None, target_name = None, kernel = None,
                        C = None):
        if train_dat != None:
            self.train_dat = train_dat
        if target_name != None:
            self.target_name = train_dat
        if kernel != None:
            self.kernel = kernel
        if C != None:
            self.C = C

        X, y = self.DatConv()
        self.model = self.fit(X, y)

    def DatConv(self):
        X = []
        for index, row in self.train_dat.iterrows():
            X.append(np.array(row.drop([self.target_name])))
        y = np.array(self.train_dat[self.target_name])
        return np.array(X), y

    def fit(self, X, y):
        thresh = 1e-5
        C = self.C

        n_samp, n_feat = X.shape
        # Compute the Gram matrix
        K = np.zeros((n_samp, n_samp))
        for i in range(n_samp):
            for j in range(n_samp):
                K[i,j] = self.kernel(X[i], X[j])
        # construct P, q, A, b, G, h matrices for CVXOPT
        P = matrix(np.outer(y,y) * K)
        q = matrix(np.ones(n_samp) * -1)
        A = matrix(y, (1, n_samp))
        b = matrix(0.0)
        if C is None:      # hard-margin SVM
            G = matrix(np.diag(np.ones(n_samp) * -1))
            h = matrix(np.zeros(n_samp))
        else:              # soft-margin SVM
            G = matrix(np.vstack((np.diag(np.ones(n_samp) * -1), \
                        np.identity(n_samp))))
            h = matrix(np.hstack((np.zeros(n_samp), np.ones(n_samp) * C)))
        # solve QP problem
        sol = qp(P, q, G, h, A, b)['x']
        
        # Construct a list of support vectors
        alphas = []

        if C is not None:
            upper_thresh = C * (1 - thresh)
        for i in range(n_samp):
            if (sol[i] > thresh):
                if (C is None) or (sol[i] < upper_thresh):
                    alphas.append(supportVector(sol[i], X[i], y[i]))

        return alphas

    def predict(self, Xout):
        n_feat = self.train_dat.shape[1] - 1

        if (n_feat != Xout.shape[1]):
            raise ValueError('Data needs correct number of attributes')

        x0 = self.model[0].X
        y0 = self.model[0].y

        b = y0
        for a in self.model:
            b -= a.alpha * a.y * self.kernel(x0, a.X)

            print '{} {} {}'.format(a.alpha, a.y, self.kernel(x0, a.X))


        y_predict = []
        for index, row in Xout.iterrows():
            yy = b
            for a in self.model:
                yy += a.alpha * a.y * self.kernel(a.X, row)

            y_predict.append(np.sign(yy))

        return y_predict




