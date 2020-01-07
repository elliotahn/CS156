import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from svmutil import *
from contour2d import *
from contour2d import draw_boundary

# CHANGE DEFAULT PLOTTING PARAMETERS
plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')

def gen_circ_data(size):
    df = pd.DataFrame((2 * np.random.random(2 * size) - 1).reshape(size, 2),
                        columns = list('xy'))
    r = np.random.random()

    df['side'] = np.sign(df['x']**2 + df['y']**2 - r**2)

    # ADD SOME NOISE
    for index in np.random.randint(0, size, size / 10):
        df.loc[index]['side'] = np.random.choice([-1, 1], 1)[0]

    return df, r

def form_conv(df, target):
    yy = df[target]
    xx = df.drop(target, axis = 1)

    return np.matrix(xx), np.array(yy)


if __name__=='__main__':
    df, r = gen_circ_data(500)

    xx, yy = form_conv(df, 'side')
    prob = svm_problem(yy, xx)
    param = svm_parameter('-s 0 -t 1 -d 2 -g 1 -c 1')
    mm = svm_train(prob, param)

    draw_boundary(df, mm, 'x', 'y', 'side', 1000, 'circdata.png')


