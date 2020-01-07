import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contour2d import *
from svmutil import *


# CHANGE DEFAULT PLOTTING PARAMETERS
plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')

import warnings
warnings.filterwarnings('ignore')

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

def form_conv(df, target):
    yy = df[target]
    xx = df.drop(target, axis = 1)

    return np.matrix(xx), np.array(yy)

# calc_error
# CALCULATE THE ERROR OF THE PREDICTIONS VERSUS THE ACTUAL. THERE ARE TWO
# TYPES OF CLASSIFICATION: ONE-VERSUS-ONE AND ONE-VERSUS-ALL. THE TYPE OF
# CLASSIFICATION IS SPECIFIED BY THE PARAMETER CLASSTYPE WHICH CAN TAKE THE
# FORM 'one' OR 'all'. THE ERROR MEASURE IS THE BINARY CLASSIFICATION ERROR.
# A CLASSIFICATION OF +1 CORRESPONDS TO NUM1, AND A CLASSIFICATION OF -1
# CORRESPONDS TO NUM2 OR OTHERS.

def calc_error(df, pred):
    total = len(pred)
    total_sum = sum(df['Number'] * pred)
 
    return float(total - total_sum) / (2. * total)

# cross_validate
# THIS FUNCTION TAKES AN INPUT DATAFRAME, A KERNEL, THE COST PENALTY C, AND
# CLASSIFICATION PARAMETERS classtype, num1, AND num2. THIS RETURNS AN
# ESTIMATION OF THE Eout, KNOWN AS THE CROSS-VALIDATION ERROR.
# --------------------------------------------------------------------------
# THE METHOD OF THIS CROSS-VALIDATION IS TO SPLIT THE DATA INTO 10 PARTITIONS
# THEN TAKE ONE PARTITION AS THE VALIDATION SET AND THE REST AS THE TRAINING
# SET AND CALCULATE THE Eout ERROR. THEN ITERATE THROUGH THE PARITIONS AS
# THE VALIDATION SET AND AVERAGE OUT THE Eout, WHOSE VALUE WILL BE RETURNED.

def cross_validate(dfin, param_str, classtype, num1, num2 = None):
    # SELECT THE APPROPRIATE SUBSET OF THE DATA BASED ON THE
    # CLASSIFICATION TYPE. THEN RANDOMIZE DATA ORDERING
    df = pick_data(dfin, 'Number', classtype, num1, num2).sample(frac = 1)

    # CALCULATE THE SIZE OF EACH SUBSET OF THE DATAFRAME WHEN SPLIT INTO
    # FOLD PARTITIONS. THE VARIABLE modulo IS TO COUNT THE NUMBER OF
    # PARTITIONS THAT WILL TAKE AN EXTRA DATAPOINT
    FOLD = 10
    modulo = df.shape[0] % FOLD
    size = df.shape[0] / FOLD
    Eout = 0.

    front = 0
    end = 0

    # SPLIT THE DATASET INTO TRAINING AND VALIDATION SETS
    for i in range(FOLD):
        front = end
        end += size
        if i < modulo:
            end += 1

        dfval = df.iloc[front:end]
        dftrain = df.iloc[range(0, front) + range(end, df.shape[0])]

        xx, yy = form_conv(dftrain, 'Number')
        prob = svm_problem(yy, xx)
        param = svm_parameter(param_str)
        mm = svm_train(prob, param)
        xt, yt = form_conv(dfval, 'Number')
        p_lab = svm_predict(yt, xt, mm, '-q')[0]

        # CALCUALTE THE CV EOUT
        Eout += calc_error(dfval, p_lab)

    return Eout / FOLD
    

def pick_data(df, pred, classtype, num1, num2 = None):
    if classtype == 'all':
        df2 = df.copy()
    else:
        df2 = df[(df[pred] == num1) | (df[pred] == num2)]

    df2.loc[df[pred] != num1, pred] = -1.
    df2.loc[df[pred] == num1, pred] = 1.

    return df2

# plotData
# GET ORIGINAL DATAFRAME AND PLOT THE DATA

def basicSV(dfinput, dftest, classtype, Q, C, num1, num2 = None):
    if classtype == 'all':
        print str(num1) + ' VS ALL\tQ = ' + str(Q) + '\tC = ' + str(C)
        suff = str(num1) + 'vsall'
    else:
        print str(num1) + ' VS ' + str(num2) + '\tQ = ' + str(Q) + \
                    '\tC = ' + str(C)
        suff = str(num1) + 'vs' + str(num2)
    suff = suff + 'q' + str(Q) + 'c' + str(C)
    df = pick_data(dfinput, 'Number', classtype, num1, num2)
    dft = pick_data(dftest, 'Number', classtype, num1, num2)
    xx, yy = form_conv(df, 'Number')
    xtest, ytest = form_conv(dft, 'Number')
    prob = svm_problem(yy, xx)
    param_str = '-s 0 -t 1 -g 1 -r 1 -q'
    param_str += ' -c ' + str(C) + ' -d ' + str(Q)
    param = svm_parameter(param_str)
    mm = svm_train(prob, param, '-q')
    p_lab = svm_predict(yy, xx, mm)[0]
    p_lab_out = svm_predict(ytest, xtest, mm, '-q')[0]
    print 'nSV : ' + str(mm.get_nr_sv())
    print 'Ein : ' + str(calc_error(df, p_lab))
    print 'Eout : ' + str(calc_error(dft, p_lab_out))

    draw_boundary(df, mm, 'Intensity', 'Symmetry', 'Number', 500,
                    'boundary' + suff + '.png')
    print '\n'

def poly_kernels(dftrain, dftest):
    param_str = '-s 0 -t 1 -g 1 -r 1 -q'

    for Q in [2, 5]:
        for C in [0.0001, 0.001, 0.01, 0.1, 1]:
            pstr = param_str
            pstr += ' -c ' + str(C) + ' -d ' + str(Q)
            EinProb(dftrain, dftest, 'one', Q, C, 1, 5)

def cross_validation_problem(dftrain, dftest):
    param_str = '-s 0 -t 1 -g 1 -r 1 -q -d 2'
    Clst = [0.0001, 0.001, 0.01, 0.1, 1]
    Cdict = dict((C, 0) for C in Clst)
    ECVlst = dict((C, 0.) for C in Clst)
    iterations = 100
    for i in range(iterations):
        print 'Iteration ' + str(i)
        ECVCount = []
        for C in Clst:
            pstr = param_str
            pstr += '-d 2 ' + '-c ' + str(C)
            ECV = cross_validate(dftrain, param_str, 'one', 1, 5)
            ECVCount.append(ECV)
            ECVlst[C] += ECV
        index_min = min(xrange(len(ECVCount)), key = ECVCount.__getitem__)
        Cdict[Clst[index_min]] += 1

    for C, ECV in Cdict.iteritems():
        print 'C = ' + str(C) + '\n' + str(ECVlst[C] / iterations) + '\n' + \
                    str(ECV) + '\n'

def RBFSV(dfinput, dftest, classtype, param_str, C, num1, num2 = None):
    if classtype == 'all':
        print str(num1) + ' VS ALL\tC = ' + str(C)
        suff = str(num1) + 'vsall'
    else:
        print str(num1) + ' VS ' + str(num2) + '\tC = ' + str(C)
        suff = str(num1) + 'vs' + str(num2)
    suff = suff + 'c' + str(C)
    df = pick_data(dfinput, 'Number', classtype, num1, num2)
    dft = pick_data(dftest, 'Number', classtype, num1, num2)
    xx, yy = form_conv(df, 'Number')
    xtest, ytest = form_conv(dft, 'Number')
    prob = svm_problem(yy, xx)
    param = svm_parameter(param_str)
    mm = svm_train(prob, param)
    p_lab = svm_predict(yy, xx, mm)[0]
    p_lab_out = svm_predict(ytest, xtest, mm, '-q')[0]
    print 'nSV : ' + str(mm.get_nr_sv())
    print 'Ein : ' + str(calc_error(df, p_lab))
    print 'Eout : ' + str(calc_error(dft, p_lab_out))

    draw_boundary(df, mm, 'Intensity', 'Symmetry', 'Number', 500,
                    'RBF' + suff + '.png')
    print '\n'

def RBF_Kernel(dftrain, dftest):
    param_str = '-s 0 -t 2 -g 1 -q'
    Clst = 10.**(np.arange(-2, 8, 2))
    for C in Clst:
        pstr = param_str
        pstr += ' -c ' + str(C)
        RBFSV(dftrain, dftest, 'one', pstr, C, 1, 5)


if __name__=='__main__':
    # WRITE MAIN CODE HERE
    dftrain = get_features('features.train.txt')
    dftest = get_features('features.test.txt')

    #poly_kernels(dftrain, dftest)
    #cross_validation_problem(dftrain, dftest)

    RBF_Kernel(dftrain, dftest)