import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from svmutil import *

plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')

import warnings
warnings.filterwarnings('ignore')

class Cluster:
    def __init__(self, K_size, df, gamma = 1., offset = False):
        self.K_size = K_size
        self.centers = None

        self.df = df.copy()
        self.df['center'] = np.zeros(df.shape[0])

        self.beta = None
        self.gamma = gamma
        self.offset = offset

        self.run_Lloyd(-1.,1.,-1.,1.)
        self.getWeights()

    def gen_centers(self, xmin, xmax, ymin, ymax):
        xc = np.random.random(self.K_size) * (xmax - xmin) + xmin
        yc = np.random.random(self.K_size) * (ymax - ymin) + ymin
        self.centers = zip(xc, yc)

    def organize_cluster(self):
        __x1, __x2, __y = self.df.columns[:3]

        for index, row in self.df.iterrows():
            min_index = 0
            min_value = np.inf
            for i in range(self.K_size):
                sqdst = (self.centers[i][0] - row[__x1])**2  \
                        + (self.centers[i][1] - row[__x2])**2
                if sqdst < min_value:
                    min_index = i
                    min_value = sqdst
            row['center'] = min_index

    def adj_centers(self):
        ischange = False
        dfcenters = self.df.groupby('center').mean()
        __x1, __x2 = self.df.columns[:2]
        for i in dfcenters.index:
            j = int(i)
            x0, y0 = self.centers[j]
            x_c = dfcenters.loc[i][__x1]
            y_c = dfcenters.loc[i][__x2]
            if (x_c != x0) or (y_c != y0):
                ischange = True
            self.centers[j] = (x_c, y_c)
        return ischange

    def check_empty_clusters(self):
        if len(self.centers) == self.df.groupby('center').count().shape[0]:
            return False
        return True

    def plot_clusters(self, fname):
        fig = sns.lmplot(x = 'x1', y = 'x2', data = self.df, fit_reg = False,
                hue = 'center', scatter_kws = {'s' : 3})
        ax = fig.axes[0][0]
        xc, yc = zip(*self.centers)
        ax.scatter(xc, yc, color = 'k', s = 10)
        plt.savefig(fname, bbox_inches = 'tight', dpi = 300)
        plt.close()

    def run_Lloyd(self, xmin, xmax, ymin, ymax):
        self.gen_centers(xmin, xmax, ymin, ymax)
        self.organize_cluster()
        while self.adj_centers():
            self.organize_cluster()

    def getexpmat(self, X):
        C = np.array(self.centers)
        func = lambda a : np.linalg.norm(X - a, axis = 1)**2
        P = np.apply_along_axis(func, 1, C).transpose()
        P = np.exp(-self.gamma * P)
        return P

    def getWeights(self):
        X = np.array(self.df[self.df.columns[:2]])
        y = self.df['y']
        P = self.getexpmat(X)
        if self.offset:
            P = np.hstack((np.ones(X.shape[0]).reshape(X.shape[0], 1), P))

        Pt = np.transpose(P)

        w = np.matmul(np.matmul(np.linalg.inv(np.matmul(Pt, P)), Pt), y)

        self.beta = np.ravel(w)

    def predict(self, X, y_pred = None):
        P = self.getexpmat(X)

        if self.offset:
            P = np.hstack((np.ones(X.shape[0]).reshape(X.shape[0], 1), P))

        p_lab = np.sign(np.ravel(np.matmul(P, self.beta)))
        
        if y_pred is None:
            return p_lab

        T = float(len(y_pred))
        D = np.dot(p_lab, y_pred)
        p_acc = (T + D) / (2. * T)

        return p_lab, p_acc


    def get_centers(self):
        return list(self.centers)





# --------------- LINEAR REGRESSION WITH REGULARIZATION -------------------

# get_features
# READ FROM A TEXT FILE AND TRANSFER THE DATA INTO A PANDAS DATAFRAME. ADD
# A __const__ COLUMN OF ONES FOR THE w0 WEIGHT.

def get_features(fname):
    col_names_initial = ['Number', 'Intensity', 'Symmetry']
    col_names = ['Intensity', 'Symmetry', 'Number']

    df = pd.read_csv(fname, sep = '\s+', header = None)
    df.columns = col_names_initial
    return df[col_names]

# form_conv
# CONVERT OUR DATAFRAME INTO A FORM THAT IS ACCEPTED BY THE SVMUTIL PACKAGE

def form_conv(df):
    label = df.columns[-1]
    yy = df[label]
    xx = df.drop(label, axis = 1)

    return np.array(xx), np.array(yy)

# pick_data
# PICK RELEVANT INFORMATION FROM THE DATAFRAME
def pick_data(df, classtype, num1, num2 = None):
    label = df.columns[-1]
    if classtype == 'all':
        df2 = df.copy()
    else:
        df2 = df[(df[label] == num1) | (df[label] == num2)]

    df2.loc[df[label] != num1, label] = -1.
    df2.loc[df[label] == num1, label] = 1.

    return df2

# calc_error
# GIVEN THE PREDICTED LABELS, CALCULATE THE BINARY CLASSIFICATION ERROR
def calc_error(df, pred):
    label = df.columns[-1]
    total = len(pred)
    total_sum = sum(df[label] * pred)
 
    return float(total - total_sum) / (2. * total)

# w_reg
# WITH THE TRADITIONAL WEIGHT DECAY REGULARIZER, MINIMIZE THE AUGMENTED ERROR
def calc_wreg(df, lam):
    label = df.columns[-1]
    X, Y = form_conv(df)
    Xt = np.transpose(X)
    lamI = lam * np.identity(X.shape[1])
    wreg = np.matmul(np.linalg.inv(np.matmul(Xt, X) + lamI), Xt)
    wreg = np.matmul(wreg, Y)
    return np.ravel(wreg)

# get_label_from_w
# GET OUR PREDICTION LABELS GIVEN THE CALCULATED WEIGHT
def get_label_from_w(X, w):
    return np.sign(np.ravel(np.matmul(X, w)))

def weight_decay_error(dfin, dfout, lam):
    Xin, Yin = form_conv(dfin)
    Xout, Yout = form_conv(dfout)
    wreg = calc_wreg(dfin, lam)
    labels_in = get_label_from_w(Xin, wreg)
    labels_out = get_label_from_w(Xout, wreg)
    Ein = calc_error(dfin, labels_in)
    Eout = calc_error(dfout, labels_out)

    print 'Ein : ' + str(Ein)
    print 'Eout : ' + str(Eout)


# add_const_col
# ADD THE CONSTANT X0 COLUMN INTO THE DATAFRAME
def add_const_col(df):
    df2 = df.copy()
    df2['__1__'] = np.ones(df.shape[0])
    col = list(df.columns)
    col.insert(0, '__1__')
    df2 = df2[col]
    return df2

# quad_transform
# APPLY A QUADRATIC TRANSFORMATION TO THE DATASET
def quad_transform(df):
    df2 = df.copy()
    df2.columns = ['x1', 'x2', 'y']
    df2['x1x2'] = df2['x1'] * df2['x2']
    df2['x1s'] = df2['x1']**2
    df2['x2s'] = df2['x2']**2
    df2 = add_const_col(df2)
    col = list(df2.columns)
    col.remove('y')
    col.append('y')
    return df2[col]

def reg_lin_regres(dftrain, dftest):
    # ALL cases
    for i in range(10):
        lam = 1.
        print '\n{} vs ALL\tlambda = {}'.format(i, lam)
        dfin = add_const_col(pick_data(dftrain, 'all', i))
        dfout = add_const_col(pick_data(dftest, 'all', i))
        weight_decay_error(dfin, dfout, lam)

        # Apply quadratic transformation
        print 'With Quadratic Transform'
        dfin = quad_transform(pick_data(dftrain, 'all', i))
        dfout = quad_transform(pick_data(dftest, 'all', i))
        weight_decay_error(dfin, dfout, lam)

    # Compare lambda = 0.1 vs 1 for 1 vs 5 classifier.
    print '\n1 VS 5 CLASSIFIER'
    dfin = quad_transform(pick_data(dftrain, 'one', 1, 5))
    dfout = quad_transform(pick_data(dftest, 'one', 1, 5))
    print 'LAMBDA = 0.01'
    weight_decay_error(dfin, dfout, 0.01)
    print 'LAMBDA = 1'
    weight_decay_error(dfin, dfout, 1.)

# --------------------- SUPPORT VECTOR MACHINES ----------------------------

def plot_support_vec_data(df):
    yy = np.linspace(-2., 2., 500)
    xx = 0.5 * yy**2 - 0.75
    fig = sns.lmplot(x = 'x1', y = 'x2', data = df, hue = 'y', 
                        fit_reg = False, scatter_kws = {'s' : 15})
    ax = fig.axes[0][0]
    ax.plot(xx, yy, color = 'k')
    plt.savefig('support_vec_plot.png', bbox_inches = 'tight', dpi = 300)
    plt.close()

def support_vec():
    df = pd.DataFrame([[1., 0., -1.], [0., 1., -1.], [0., -1., -1.], \
        [-1., 0., 1.], [0., 2., 1.], [0., -2., 1.], [-2., 0., 1.]], \
                        columns = ['x1', 'x2', 'y'])
    plot_support_vec_data(df)

    xx, yy = form_conv(df)
    prob = svm_problem(yy, xx)
    param = svm_parameter('-s 0 -t 1 -g 1 -r 1 -d 2')
    mm = svm_train(prob, param)
    print 'Num SVM : {}'.format(mm.get_nr_sv())
    ind = mm.get_sv_indices()
    sv_lst = mm.get_sv_coef()
    print sv_lst
    print ind

def gen_square_data(size):
    df = pd.DataFrame((np.random.random(2 * size) * 2 - 1).reshape(size, 2),\
                        columns = ['x1', 'x2'])
    df['y'] = np.sign(df['x2'] - df['x1'] + 0.25 * np.sin(np.pi * df['x1']))
    return df

def plot_data(df, fname, Mtype, m = None):
    fig = sns.lmplot(x = 'x1', y = 'x2', data = df, hue = 'y',
                        fit_reg = False, scatter_kws = {'s' : 3})
    ax = fig.axes[0][0]
    if m is not None:
        N = 1000
        xx = np.linspace(-1, 1, N)
        yy = np.linspace(-1, 1, N)
        mat = np.vstack((np.repeat(xx, N), np.tile(yy, N))).transpose()
        if Mtype == 'SVM':
            p_lab = svm_predict([], mat, m, '-q')[0]
        if Mtype == 'RBF':
            p_lab = m.predict(mat, None)
        label = np.array(p_lab).reshape(N, N)
        Y, X = np.meshgrid(yy, xx)
        ax.contour(X, Y, label, [0], linewidths = 1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.savefig(fname, bbox_inches = 'tight', dpi = 300)
    plt.close()

def RBF_SVM(df, gamma):
    xx, yy = form_conv(df)
    prob = svm_problem(yy, xx)
    param = svm_parameter('-s 0 -t 2 -c 1e5 -q -g ' + str(gamma))
    mm = svm_train(prob, param)

    return mm

def RBF_SVM_Predict(mm, X, ylabel):
    p_lab, p_acc, p_val = svm_predict(ylabel, X, mm, '-q')
    if len(ylabel) == 0:
        return p_lab
    return p_lab, p_acc[0] / 100.

def RBFProblem(gamma, K_size, fsvm, frbf, frbfc):
    size = 100
    iterations = 1000
    dfout = gen_square_data(5000)
    Xout, Yout = form_conv(dfout)

    print 'Gamma = {}\tK_size = {}\n'.format(gamma, K_size)

    # SVM SECTION
    num_sep = 0
    num_sep_rbf = 0
    eff_iter = 0
    SVMoverRBF = 0
    Eout = 0.
    Ein = 0.
    for i in range(iterations):
        dfin = gen_square_data(size)
        Xin, Yin = form_conv(dfin)
        mm = RBF_SVM(dfin, gamma)
        if RBF_SVM_Predict(mm, Xin, Yin)[1] == 1:
            RBFC = Cluster(K_size, dfin, gamma, offset = True)
            if not RBFC.check_empty_clusters():
                eff_iter += 1
                ESVMout = 1 - RBF_SVM_Predict(mm, Xout, Yout)[1]
                ERBFout = 1 - RBFC.predict(Xout, Yout)[1]
                Ein += 1 - RBFC.predict(Xin, Yin)[1]
                Eout += ERBFout
                if Ein == 0:
                    num_sep_rbf += 1
                if ESVMout < ERBFout:
                    SVMoverRBF += 1
            num_sep += 1

    print 'SVM Separable Percentage: {}'.format(float(num_sep) / iterations)
    print 'RBF Separable Percentage: {}'.format(float(num_sep_rbf) / eff_iter)
    print 'SVM over RBF : {}'.format(float(SVMoverRBF) / eff_iter)
    print 'Avg Ein for RBF Clusters : {}'.format(Ein / eff_iter)
    print 'Avg Eout for RBF Clusters : {}'.format(Eout / eff_iter)


    RBFC.plot_clusters(frbfc)
    plot_data(dfout, fsvm, 'SVM', mm)
    plot_data(dfout, frbf, 'RBF', RBFC)

    


if __name__ == '__main__':
    #dftrain = get_features('features.train.txt')
    #dftest = get_features('features.test.txt')

    #reg_lin_regres(dftrain, dftest)
    #support_vec()

    RBFProblem(1.5, 9, 'RBFPlot1.png', 'RBFContour1.png', 'clusterplot1.png')
    RBFProblem(1.5, 12, 'RBFPlot2.png', 'RBFContour2.png', 'clusterplot2.png')
    RBFProblem(2., 9, 'RBFPlot3.png', 'RBFContour3.png', 'clusterplot3.png')


    