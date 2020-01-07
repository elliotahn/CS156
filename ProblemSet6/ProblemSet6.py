import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')

def readFile(filename):
    df = pd.read_csv(filename, sep = '  ', header = None, engine = 'python')
    df.columns = ['x1', 'x2', 'output']
    df['const'] = pd.Series(np.ones(df.shape[0]))
    return df[['const', 'x1', 'x2', 'output']]

def nonLinearTrans(df):
    df['x12'] = df['x1']**2
    df['x22'] = df['x2']**2
    df['x1x2'] = df['x1'] * df['x2']
    df['|x1-x2|'] = np.abs(df['x1'] - df['x2'])
    df['|x1+x2|'] = np.abs(df['x1'] + df['x2'])
    return df[['const', 'x1', 'x2', 'x12', 'x22', 'x1x2', '|x1-x2|', \
                    '|x1+x2|', 'output']]

def lineFit(df):
    lm = LinearRegression(fit_intercept = False)
    X = df.drop('output', axis = 1)
    Y = df['output']
    lm.fit(X, Y)
    return lm.coef_

def regLineFit(df, lam):
    X = df.drop('output', axis = 1)
    Y = df['output']
    Xt = np.transpose(X)
    lamI = lam * np.identity(X.shape[1])
    wreg = np.matmul(np.linalg.inv(np.matmul(Xt, X) + lamI), Xt)
    wreg = np.matmul(wreg, Y)
    return wreg


def CalcError(df, w):
    tot_bad = 0.
    NN = df.shape[0]
    for i in range(NN):
        if np.sign(np.dot(w, df.drop('output', axis = 1).iloc[i])) != \
                    df.iloc[i]['output']:
            tot_bad += 1

    return tot_bad / NN


def weightDecay():
    df_train = nonLinearTrans(readFile('in.dta.txt'))
    df_test = nonLinearTrans(readFile('out.dta.txt'))

    print 'LINEAR REGRESSION'
    wlin = lineFit(df_train)
    print 'Ein: ' + str(CalcError(df_train, wlin))
    print 'Eout ' + str(CalcError(df_test, wlin))

    for k in range(-5, 6):
        print '\nREGULARIZATION WITH k = ' + str(k)
        wreg = regLineFit(df_train, 10**k)
        print 'Ein: ' + str(CalcError(df_train, wreg))
        print 'Eout ' + str(CalcError(df_test, wreg))



if __name__=='__main__':
    weightDecay()