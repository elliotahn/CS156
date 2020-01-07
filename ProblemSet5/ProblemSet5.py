import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')

def Euv(u, v):
    return (u * np.exp(v) - 2 * v * np.exp(-u))**2

def DuEuv(u, v):
    return 2 * (u * np.exp(v) - 2 * v * np.exp(-u)) \
            * (np.exp(v) + 2 * v * np.exp(-u))

def DvEuv(u, v):
    return 2 * (u * np.exp(v) - 2 * v * np.exp(-u)) \
            * (u * np.exp(v) - 2 * np.exp(-u))

def GradDescent(eta):
    u, v = 1, 1

    iteration = 0
    while (Euv(u, v) >= 1e-14):
        iteration += 1
        du = - eta * DuEuv(u, v)
        dv = - eta * DvEuv(u, v)
        u += du
        v += dv

    print iteration
    print u, v
    print Euv(u, v)

    u, v = 1, 1

    iteration = 0
    while iteration < 15:
        iteration += 1
        u -= eta * DuEuv(u, v)
        v -= eta * DvEuv(u, v)

    print Euv(u, v)

def genLine():
    x1, x2 = np.random.random(2) * 2 - 1
    y1, y2 = np.random.random(2) * 2 - 1
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def getData(size):
    df = pd.DataFrame((np.random.random(2 * size) * 2 - 1).reshape(size, 2),
                        columns = ['x', 'y'])
    df['const'] = np.ones(size)
    return df[['const', 'x', 'y']]

def calcSides(df, m, b):
    df['side'] = np.sign(df['y'] - m * df['x'] - b)

def logFunc(row, w):
    s = np.dot(row, w)
    return 2 / (1 + np.exp(-s)) - 1

def plotData(df, m, b):
    xx = np.linspace(-1.1, 1.1, 10)
    yy = m * xx + b
    df.plot.scatter(x = 'x', y = 'y', c = 'side', cmap = 'coolwarm')
    plt.plot(xx, yy, 'black')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.savefig('scatterplotset5.png', bbox_inches = 'tight', dpi = 500)
    plt.close()

def plotResults(df, m, b):
    xx = np.linspace(-1.1, 1.1, 10)
    yy = m * xx + b
    df.plot.scatter(x = 'x', y = 'y', c = 'pred', cmap = 'coolwarm')
    plt.plot(xx, yy, 'black')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.savefig('scatterpredset5.png', bbox_inches = 'tight', dpi = 500)
    plt.close()

def calcPredictions(df, w):
    f = lambda row : 2 / (1 + np.exp(-np.dot(row, w))) - 1
    df['pred'] = df.apply(lambda row : f(row[['const', 'x', 'y']]),
                                axis = 1)

def logresSGD(df, eta):
    w = np.zeros(len(df.columns) - 1)
    result = df.columns[-1]
    NN = df.shape[0]
    lst = range(NN)
    epoch = 0

    while True:
        wnew = np.zeros(len(df.columns) - 1) + w
        if epoch % 10 == 0:
            print 'epoch : ' + str(epoch)
        epoch += 1
        np.random.shuffle(lst)
        for i in lst:
            row = df.iloc[i]
            yn = row[result]
            dw = yn * row[:-1] / (1 + np.exp(yn * np.dot(wnew, row[:-1])))
            wnew += dw * eta

        if (np.linalg.norm(wnew - w) < 0.01):
            w = wnew
            break
        w = wnew

    return w, epoch

def calcEout(df, w, m, b):
    if 'side' not in df.columns:
        calcSides(df, m, b)
    err = 0.

    for index, row in df.iterrows():
        yn = row['side']
        expterm = -yn * np.dot(w, row[['const', 'x', 'y']])
        err += np.log(1 + np.exp(expterm))

    return err / df.shape[0]

def LogResProblem():

    # GET DATA
    print 'Getting Data'
    m, b = genLine()
    df = getData(100)
    calcSides(df, m, b)
    plotData(df, m, b)

    # PERFORM LOGISTIC REGRESSION
    print 'Fitting Data'
    w, epoch = logresSGD(df, 0.01)
    calcPredictions(df, w)

    # CALCULATE EOUT
    print 'Calculating Eout'
    df2 = getData(1000)
    calcPredictions(df2, w)
    Eout = calcEout(df2, w, m, b)

    plotResults(df2, m, b)

    return Eout, epoch

if __name__=='__main__':
    #GradDescent(0.1)
    iterations = 10
    Eoutmean = 0.
    epochmean = 0.
    for i in range(iterations):
        print 'Iteration ' + str(i)
        Eout, epoch = LogResProblem()
        Eoutmean += Eout
        epochmean += epoch
    print Eoutmean / iterations, epochmean / iterations
