import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SVMPython import *
from svmutil import *
import time
from contour2d import draw_boundary

# CHANGE DEFAULT PLOTTING PARAMETERS
plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')



def gen_data(size):
    df = pd.DataFrame((2 * np.random.random(2 * size) - 1).reshape(size, 2),
                        columns = list('xy'))
    m, b = gen_line()
    df['side'] = np.sign(df['y'] - m * df['x'] - b)

    for index in np.random.randint(0, size, size / 10):
        df.loc[index]['side'] = np.random.choice([-1, 1], 1)[0]

    return df, m, b

def gen_line():
    x1, x2, y1, y2 = 2 * np.random.random(4) - 1
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b    

def plot_data(df, m, b):
    xplot = np.linspace(-1, 1, 100)
    yplot = m * xplot + b
    sns.lmplot(x = 'x', y = 'y', data = df, hue = 'side', fit_reg = False,
                scatter_kws = {'s' : 3})
    plt.plot(xplot, yplot, linewidth = 1, color = 'k')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig('testdata.png', bbox_inches = 'tight', dpi = 500)
    plt.close()


def form_conv(df, target):
    yy = df[target]
    xx = df.drop(target, axis = 1)

    return np.matrix(xx), np.array(yy)


if __name__=='__main__':
    df, m, b = gen_data(500)
    plot_data(df, m, b)

    xx, yy = form_conv(df, 'side')

    prob = svm_problem(np.array(yy), np.matrix(xx))
    param = svm_parameter('-s 0 -t 0 -c 0.1 ')
    mm = svm_train(prob, param)

    #p_lab, p_acc, p_val = svm_predict(yy, xx, mm)

    density = 1000
    #draw_boundary(df, mm, 'x', 'y', density)

    #wishing girl lola marsh

