import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from svmutil import *

# CHANGE DEFAULT PLOTTING PARAMETERS
plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')

def draw_boundary(df, model, xname, yname, target, density, filename):
    # SET UP COORDINATE SYSTEM FOR PLOTTING BY KNOWING THE MIN AND MAX VALUES
    xmin = min(df[xname])
    xmax = max(df[xname])
    ymin = min(df[yname])
    ymax = max(df[yname])

    dx = (xmax - xmin) / (density - 1)
    dy = (ymax - ymin) / (density - 1)

    # CREATE A GRID OF THE X-Y SPACE AND FORMAT INTO A MATRIX
    xx = np.linspace(xmin, xmax, density)
    yy = np.linspace(ymin, ymax, density)

    mat = []
    for x in xx:
        for y in yy:
            mat.append([x, y])
    mat = np.matrix(mat)

    # GET PREDICTION LABELS ON OUR GRID
    p_lab, p_acc, p_val = svm_predict([], mat, model, '-q')

    Y, X = np.meshgrid(yy, xx)

    # RESHAPE THE PREDICTION LABELS INTO A 2D GRID
    label = np.array(p_lab).reshape(density, density)

    plt.figure()
    bplot = sns.lmplot(x = xname, y = yname, data = df, hue = target, 
                        fit_reg = False, scatter_kws = {'s' : 3})
    ax = bplot.axes[0, 0]
    ax.contour(X, Y, label, [0], linewidths = 1)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.savefig(filename, dpi = 500)
    plt.close()

def draw_region(df, model, xname, yname, target, density):
    # SET UP COORDINATE SYSTEM FOR PLOTTING BY KNOWING THE MIN AND MAX VALUES
    xmin = min(df[xname])
    xmax = max(df[xname])
    ymin = min(df[yname])
    ymax = max(df[yname])

    dx = (xmax - xmin) / (density - 1)
    dy = (ymax - ymin) / (density - 1)

    # CREATE A GRID OF THE X-Y SPACE AND FORMAT INTO A MATRIX
    xx = np.linspace(xmin, xmax, density)
    yy = np.linspace(ymin, ymax, density)

    mat = []
    for x in xx:
        for y in yy:
            mat.append([x, y])
    mat = np.matrix(mat)

    # GET PREDICTION LABELS ON OUR GRID
    p_lab, p_acc, p_val = svm_predict([], mat, model, '-q')

    df2 = pd.DataFrame(mat, columns = [xname, yname])
    df2[target] = p_lab
    sns.lmplot(x = xname, y = yname, data = df2, hue = target,
                fit_reg = False, scatter_kws = {'s' : 3})
    plt.savefig('regionplot.png', dpi = 500)
    plt.close()
