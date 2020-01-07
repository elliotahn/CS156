# IMPORT NUMPY, PANDAS, SCIPY, SKLEARN, AND PLOTTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
from scipy.optimize import minimize

# IMPORT CVXOPT LIBRARIES
from cvxopt import matrix
from cvxopt.solvers import qp
from SVMPython import SVMQuadProg, SVMQuadProg2

# IGNORE WARNINGS
import warnings
warnings.filterwarnings('ignore')


# CHANGE DEFAULT PLOTTING PARAMETERS
plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')

# VALIDATION PROBLEMS 1-5
def readFile(filename):
    df = pd.read_csv(filename, sep = '  ', header = None, engine = 'python')
    df.columns = ['x1', 'x2', 'output']
    df['__const__'] = pd.Series(np.ones(df.shape[0]))
    return df[['__const__', 'x1', 'x2', 'output']]

def nonLinearTrans(df):
    df['x12'] = df['x1']**2
    df['x22'] = df['x2']**2
    df['x1x2'] = df['x1'] * df['x2']
    df['|x1-x2|'] = np.abs(df['x1'] - df['x2'])
    df['|x1+x2|'] = np.abs(df['x1'] + df['x2'])
    return df[['__const__', 'x1', 'x2', 'x12', 'x22', 'x1x2', '|x1-x2|', \
                    '|x1+x2|', 'output']]

def lineFit(df):
    lm = LinearRegression(fit_intercept = False)
    X = df.drop('output', axis = 1)
    Y = df['output']
    lm.fit(X, Y)
    return lm.coef_

def CalcError(df, w):
    tot_bad = 0.
    NN = df.shape[0]
    for i in range(NN):
        if np.sign(np.dot(w, df.drop('output', axis = 1).iloc[i])) != \
                    df.iloc[i]['output']:
            tot_bad += 1

    return tot_bad / NN

def validation():

    # PROBLEM ONE
    dfin = nonLinearTrans(readFile('in.dta.txt'))
    dfout = nonLinearTrans(readFile('out.dta.txt'))

    dftrain = dfin[:25]
    dfval = dfin[25:]

    size = len(dftrain.columns)

    # Run through each model and print classification error
    print 'LEARNING FROM TRAINING SET'
    for i in range(1, size):
        print 'k = ' + str(i - 1)
        w = lineFit(dftrain.drop(dftrain.columns[i:size - 1], axis = 1))
        print CalcError(dfval.drop(dfval.columns[i:size - 1], axis = 1),\
                            w)
        print CalcError(dfout.drop(dfout.columns[i:size - 1], axis = 1),\
                            w)

    print 'LEARNING FROM VALIDATION SET'
    for i in range(1, size):
        print 'k = ' + str(i - 1)
        w = lineFit(dfval.drop(dfval.columns[i:size - 1], axis = 1))
        print CalcError(dftrain.drop(dftrain.columns[i:size - 1], axis = 1),\
                            w)
        print CalcError(dfout.drop(dfout.columns[i:size - 1], axis = 1),\
                            w)


# PLA VS SVM PROBLEMS

def genLine():
    x1, x2 = np.random.random(2) * 2 - 1
    y1, y2 = np.random.random(2) * 2 - 1
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def getData(size):
    df = pd.DataFrame((np.random.random(2 * size) * 2 - 1).reshape(size, 2),
                        columns = ['x', 'y'])
    df['__const__'] = np.ones(size)
    return df[['__const__', 'x', 'y']]

def calcSides(df, m, b):
    df['side'] = np.sign(df['y'] - m * df['x'] - b)

def getSplitData(size):
    # This is getting a set of points in X in [-1, 1]^2 with a target
    # function but all points cannot lie on one side of the target line.

    while True:
        df = getData(size)
        m, b = genLine()
        calcSides(df, m, b)

        oneside = df['side'].value_counts().iloc[0]
        if not ((oneside == df.shape[0]) or (oneside == 0)):
            return df, m, b


def plotData(df, m, b, w, filename):
    xx = np.linspace(-1.1, 1.1, 10)
    yy = m * xx + b
    df.plot.scatter(x = 'x', y = 'y', c = 'side', cmap = 'coolwarm')
    plt.plot(xx, yy, 'black', linewidth = 0.4)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])

    if w is not None:
        mm = -w[1] / w[2]
        bb = -w[0] / w[2]
        yw = mm * xx + bb
        plt.plot(xx, yw, 'blue', linewidth = 0.4)
        plt.fill_between(xx, yy, yw, color='grey', alpha=0.5)

    plt.savefig(filename, bbox_inches = 'tight', dpi = 500)
    plt.close()

def linef(x, m1, m2, b1, b2):
    top = max(m1 * x + b1, m2 * x + b2)
    bot = min(m1 * x + b1, m2 * x + b2)
    top = min(top, 1.)
    bot = max(bot, -1.)
    if top < -1 or bot > 1:
        return 0
    return top - bot

def calcEout(w, m, b):
    mm = - w[1] / w[2]
    bb = - w[0] / w[2]

    return quad(linef, -1., 1., args = (m, mm, b, bb))[0] / 4.

def checkPointsPLA(df, w):
    for i in range(df.shape[0]):
        if (np.sign(np.dot(w, df.drop('side', axis = 1).iloc[i])) != \
                    df.iloc[i]['side']):
            return i
    return -1

def runPLA(df):
    w = np.zeros(3)

    while True:
        index = checkPointsPLA(df, w)
        if index == -1:
            return w
        row = df.iloc[index]
        w = w + row['side'] * row[['__const__', 'x', 'y']]
    
def plasvmerr(size):
    df, m, b = getSplitData(size)

    '''wPLA = runPLA(df)
    plotData(df, m, b, wPLA, 'PLAset7.png')
    PLAerr = calcEout(wPLA, m, b)
    print 'Eout for PLA : ' + str(PLAerr)

    wSVM = SVMQuadProg(df, 'side')
    plotData(df, m, b, wSVM, 'SVMset7.png')
    SVMerr = calcEout(wSVM, m, b)
    print 'Eout for SVM : ' + str(SVMerr)'''

    wPLA = runPLA(df)
    plotData(df, m, b, wPLA, 'PLAset7.png')
    PLAerr = calcEout(wPLA, m, b)

    wSVM, SVcount = SVMQuadProg2(df, 'side')
    plotData(df, m, b, wSVM, 'SVMset7.png')
    SVMerr = calcEout(wSVM, m, b)

    return PLAerr, SVMerr, SVcount

def plavssvm():
    nruns = 10
    size = 25
    PLA = 0
    SVM = 0
    SVav = []
    for i in range(nruns):
        print 'Iteration ' + str(i)
        PLAerr, SVMerr, SVcount = plasvmerr(size)
        SVav.append(SVcount)
        print PLAerr, SVMerr
        if (PLAerr == SVMerr):
            pass
        elif (PLAerr < SVMerr):
            PLA += 1
        else:
            SVM += 1

    print float(SVM) / (SVM + PLA), np.mean(SVav)



if __name__=='__main__':
    #validation()

    plavssvm()
