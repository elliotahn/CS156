import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')

def logGrowth(N, d):
	if N > d:
		return d * np.log(N)
	return N * np.log(2.)

def logdelta(N, params):
	d, epsilon, delta = params
	return np.log(4) + logGrowth(2 * N, d) - epsilon**2 * N / 8 \
				- np.log(delta)

def VCBound(N, delta, d):
	logterm = np.log(4.) + logGrowth(2 * N, d) - np.log(delta)
	return np.sqrt(8. / N * logterm)

def RPBound(N, delta, d):
	term1 = np.sqrt(2 * (np.log(2.) + np.log(N) + logGrowth(N, d)) / N)
	term2 = np.sqrt(- 2 / N * np.log(delta))
	return term1 + term2 + 1. / N

def PVBound(N, delta, d):
	logTerm = np.log(6.) + logGrowth(2 * N, d) - np.log(delta)
	parlst = [N]
	return findZeroN(lambda ep, param : ep**2 - 2 * ep / param[0] - \
									logTerm / param[0], parlst, 1e-5, 0.)
def DeBound(N, delta, d):
	logTerm = np.log(4.) + logGrowth(N**2, d) - np.log(delta)
	parlst = [N]
	return findZeroN(lambda ep, param : ep**2 - 1 / (2. * param[0]) * \
						(4 * ep * (1 + ep) + logTerm), parlst, 1e-5, 0.)

def findZeroN(func, params, prec, lower):
	upper_bound = 999999999
	x = lower
	sign = np.sign(func(lower, params))
	while x < upper_bound:
		if sign != np.sign(func(x, params)):
			break
		x += prec
	return x

def ploterrorN(Nlow, Nhigh, delta, d, filename):
	N = range(Nlow, Nhigh + 1)
	epVC = []
	epRP = []
	epPV = []
	epDe = []

	for n in N:
		print 'Calculating N = ' + str(n)
		epVC.append(VCBound(n, delta, d))
		epRP.append(RPBound(n, delta, d))
		epPV.append(PVBound(n, delta, d))
		epDe.append(DeBound(n, delta, d))


	plt.plot(N, epVC, label = 'VC Bound')
	plt.plot(N, epRP, label = 'RP Bound')
	plt.plot(N, epPV, label = 'PV Bound')
	plt.plot(N, epDe, label = 'De Bound')
	plt.xlabel('N')
	plt.ylabel('Generalization Error')
	plt.xlim([Nlow, Nhigh])
	plt.title('Epsilon vs N for delta = ' + str(delta) + ' and d = ' + str(d))
	plt.legend()
	plt.savefig('epsilonvssmallN.png', bbox_inches = 'tight', dpi = 900)
	plt.close()

def GenError():
	print 'Calculating Sample Size for d = 10, epsilon = 0.05, delta = 0.05'
	print 'Result: ' + str(findZeroN(logdelta, (10, 0.05, 0.05), 1, 1))

	print'\nPlotting Bounds'
	ploterrorN(1000, 10000, 0.05, 50, 'epsilonvsN.png')
	ploterrorN(3, 15, 0.05, 50, 'epsilonvssmallN.png')

def genSineData(size):
	x = np.random.random(size) * 2 - 1
	y = np.sin(np.pi * x)
	df = pd.DataFrame()
	df['x'] = pd.Series(x)
	df['y'] = pd.Series(y)
	return df

def LinRegression(df):
	# BASIC LINEAR REGRESSION WHERE BETA = (Xt X)^-1 Xt Y
	X = df['x']
	Y = df['y']
	return 1. / np.dot(X, X) * np.dot(X, Y)

def LineFit(df, cols):
	lm = LinearRegression(fit_intercept = False)
	X = df[cols]
	Y = df['sine']
	lm.fit(X, Y)
	w = lm.coef_
	return w

def SinLinReg():
	print '\nTRAINING y = ax on y = sin pi x Set\n'
	alst = []
	iterations = 100

	plt.figure()
	x = np.linspace(-1, 1, 1000)
	y = np.sin(np.pi * x)

	for i in range(iterations):
		df = genSineData(2)
		a = LinRegression(df)
		plt.plot(x, a * x, 'grey', alpha = 0.3, linewidth = 0.5)
		alst.append(a)

	ahat = np.mean(alst)
	varahat = np.std(alst)**2


	# MAKE THEORETICAL CALCULATIONS
	f = lambda x1, x2: (x1 * np.sin(np.pi * x1) + x2 * np.sin(np.pi * x2)) \
							/ (x1**2 + x2**2)
	realahat = dblquad(f, -1, 1, lambda x: 0, lambda x: 1)[0] / 2
	g = lambda x1, x2: ((x1 * np.sin(np.pi * x1) + x2 * np.sin(np.pi * x2)) \
						 / (x1**2 + x2**2) - realahat)**2
	realvara = dblquad(g, -1, 1, lambda x: 0, lambda x: 1)[0] / 2

	plt.plot(x, y, 'r')
	plt.xlim([-1, 1])
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title(r'$\hat a = $' + str(round(ahat, 2)) + \
				r'  $Var \, \hat a = $' + str(round(realvara, 4)))
	plt.savefig('SinLinReg.png', dpi = 900)
	plt.close()

	print 'ahat = ' + str(ahat)
	print 'var ahat = ' + str(varahat)
	print 'realahat = ' + str(realahat)
	print 'real var ahat = ' + str(realvara)

	bias = quad(lambda x: (np.sin(np.pi * x) - ahat * x)**2, -1, 1)[0] / 2
	print '\nbias = ' + str(round(bias, 2))
	variance = realvara / 3
	print 'variance = ' + str(round(variance, 2))

def getPoints(size):
	df = pd.DataFrame()
	df['const'] = pd.Series(np.ones(size))
	df['x'] = pd.Series(np.random.random(size) * 2 - 1)
	df['x2'] = df['x'].apply(lambda a : a**2)
	df['sine'] = df['x'].apply(lambda a : np.sin(np.pi * a))
	return df

def ff(y):
	return np.sin(np.pi * y)

def calcEout():
	print 'CALCULATING EOUT FOR h = b, ax, ax + b, ax2, ax2 + b'
	iterations = 1000
	eout = np.zeros(5)
	xx = np.linspace(-1, 1, 1000)
	blst = []
	a1lst = []
	a2lst = []
	alinlst = []
	blinlst = []
	aquadlst = []
	bquadlst = []

	for i in range(iterations):
		df = getPoints(2)

		blst.append(LineFit(df, ['const'])[0])
		a1lst.append(LineFit(df, ['x'])[0])
		wlin = LineFit(df, ['const', 'x'])
		alinlst.append(wlin[0])
		blinlst.append(wlin[1])
		a2lst.append(LineFit(df, ['x2'])[0])
		wquad = LineFit(df, ['const', 'x2'])
		aquadlst.append(wquad[0])
		bquadlst.append(wquad[1])

		eout[0] += quad(lambda y: (blst[-1] - ff(y))**2 / 2, -1, 1)[0]
		eout[1] += quad(lambda y: (a1lst[-1] * y - ff(y))**2 / 2, -1, 1)[0]
		eout[2] += quad(lambda y: \
					(alinlst[-1] + blinlst[-1] * y - ff(y))**2 / 2, -1, 1)[0]
		eout[3] += quad(lambda y: (a2lst[-1] * y**2 - ff(y))**2 / 2, -1, 1)[0]
		eout[4] += quad(lambda y: (aquadlst[-1] + bquadlst[-1] * y**2 - \
									ff(y))**2 / 2, -1, 1)[0]

	b = np.mean(blst)
	a1 = np.mean(a1lst)
	alin = np.mean(alinlst)
	blin = np.mean(blinlst)
	a2 = np.mean(a2lst)
	aquad = np.mean(aquadlst)
	bquad = np.mean(bquadlst)

	for i in range(len(eout)):
		eout[i] /= iterations

	print eout





if __name__ == '__main__':
	#GenError()
	#SinLinReg()
	calcEout()

