import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['patch.force_edgecolor'] = True
sns.set_style('darkgrid')

# -------------- HOEFFDING INEQUALITY -----------------------
def simulatecoins(SIZE, numflip):
	# FLIP SIZE COINS numflip TIMES AND AVERAGE OUT
	coins = np.zeros(SIZE)
	for i in range(numflip):
		coins += np.random.randint(0, 2, size = SIZE)
	return coins / 10.

def coin_analysis(SIZE, numflip, iterations):
	first = []
	randcoin = []
	mincoin = []

	# GET DATA FOR ALL 3 COINS
	for i in range(iterations):
		coins = simulatecoins(SIZE, numflip)
		randindex = np.random.randint(0, SIZE)
		first.append(coins[0])
		randcoin.append(coins[randindex])
		mincoin.append(np.min(coins))

	# ORGANIZE DATA INTO A PANDAS DATAFRAME
	coindf = pd.DataFrame()
	coindf['First Coin'] = pd.Series(first)
	coindf['Random Coin'] = pd.Series(randcoin)
	coindf['Minimum Coin'] = pd.Series(mincoin)

	# CALCULATE AVERAGES
	firstav = coindf['First Coin'].mean()
	randav = coindf['Random Coin'].mean()
	minav = coindf['Minimum Coin'].mean()

	# SET BINS FOR HISTOGRAM
	bins = np.arange(11) / 10.

	# PLOT HISTOGRAMS FOR ALL 3 COINS
	sns.distplot(coindf['First Coin'], kde = False)
	plt.xlim([0,1])
	plt.title('Mean = ' + str(firstav))
	plt.savefig('firstcoinhist.png', bbox_inches = 'tight', bins = bins)
	plt.close()

	sns.distplot(coindf['Random Coin'], kde = False)
	plt.xlim([0,1])
	plt.title('Mean = ' + str(randav))
	plt.savefig('randcoinhist.png', bbox_inches = 'tight', bins = bins)
	plt.close()

	sns.distplot(coindf['Minimum Coin'], kde = False)
	plt.title('Mean = ' + str(minav))
	plt.savefig('mincoinhist.png', bbox_inches = 'tight', bins = 10)
	plt.close()

# --------------- LINEAR REGRESSION ---------------------

def getData(size, m, b):
	# CREATE DATA FOR POINTS IN X-SPACE
	df = pd.DataFrame((np.random.random(2 * size) * 2 - 1)\
			.reshape(size, 2), columns = ['x', 'y'])
	df['const'] = pd.Series(np.ones(size))
	df['side'] = df.apply(lambda coord : pickSide(coord, m, b), axis = 1)
	return df[['const', 'x', 'y', 'side']]

def pickLine():
	# PICK A LINE THAT DIVIDES [-1, 1] x [-1, 1] SPACE
	p = np.random.random(4) * 2 - 1

	m = (p[3] - p[2]) / (p[1] - p[0])

	b = p[2] - m * p[0]

	return m, b

def pickSide(coord, m, b):
	# GIVEN A COORDINATE AND LINE, RETURN +1 IF POINT IS ABOVE THE LINE
	return int(np.sign(coord['y'] - m * coord['x'] - b))

def checkPoints(df, w):
	# COUNT NUMBER OF POINTS THAT ARE ON THE WRONG SIDE OF THE LINE
	bad_points = 0
	for i in range(df.shape[0]):
		if (np.sign(np.dot(w, df.drop('side', axis = 1).iloc[i])) != \
					df.iloc[i]['side']):
			bad_points += 1.
	return bad_points / df.shape[0]

def plotData(df, m, y0, w, filename):
	# CREATE BOUNDARY AND ARRAY FOR REAL WEIGHTS
	plotlo = -1.2
	plothi = 1.2
	xx = np.linspace(plotlo, plothi, 1000)

	plotLine = (m is not None) and (y0 is not None)
	if plotLine:
		yy = m * xx + y0

	# SCATTER PLOT OF THE DATA AND REAL WEIGHTS LINE
	df.plot.scatter(x = 'x', y = 'y', c = 'side', cmap = 'coolwarm',
						colorbar = False)
	plt.xlim([plotlo, plothi])
	plt.ylim([plotlo, plothi])
	if plotLine:
		plt.plot(xx, yy, 'k', label='True Divider')

	# PLOT THE WEIGHTS AND FILL THE ERROR REGION
	if w is not None:
		yreg = - (w[0] + w[1] * xx) / w[2]
		plt.plot(xx, yreg, label='Regression Divider')
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		if plotLine:
			plt.fill_between(xx, yy, yreg, color='grey', alpha=0.4)

	if filename is not None:
		plt.savefig(filename, bbox_inches = 'tight', dpi = 900)
	plt.close()

def sklearnVersion(df):
	# CREATE LINEARREGRESSION OBJECT AND FIT THE DATA
	lm = LinearRegression()
	X = df.drop('side', axis = 1)
	Y = df['side']
	lm.fit(X, Y)
	w = lm.coef_
	w[0] = lm.intercept_
	return w

def manualVersion(df):
	# BASIC LINEAR REGRESSION WHERE BETA = (Xt X)^-1 Xt Y
	X = df.rename(columns = {x:y for x,y in zip(df.columns, \
						range(0,len(df.columns)))})
	X.drop(X.columns[-1], axis = 1, inplace = True)
	Y = df['side'].rename({'side' : 0})
	XT = np.transpose(X)
	w = np.matmul(np.matmul(np.linalg.inv(np.matmul(XT, X)), XT), Y)
	return w

def RegExercise():
	# PICK A RANDOM LINE
	m, b = pickLine()

	# WAV WILL BE THE AVERAGE WEIGHTS OVER CERTAIN NUMBER OF ITERATIONS
	# ERRAV IS THE AVERAGE ERROR THAT THE LINEAR REGRESSION MODEL GIVES
	wav = np.zeros(3)
	errav = 0.
	iterations = 1000

	# GENERATE 100 POINTS FOR EACH ITERATION, RUN A LINEAR REGRESSION AND 
	# AVERAGE OUT THE WEIGHTS AND CALCULATE THE AVERAGE ERROR
	print 'GETTING AVERAGE WEIGHT FOR ' + str(iterations) + ' ITERATIONS'
	for i in range(iterations):
		if i % 20 == 0:
			print 'Iteration ' + str(i)

		df = getData(100, m, b)
		w = manualVersion(df)
		wav += w
		errav += checkPoints(df, w)

	# DIVIDE BY NUMBER OF ITERATIONS FOR THE AVERAGE
	wav = wav / iterations
	errav /= iterations

	# NOW GENERATE 1000 RANDOM POINTS FOR EACH ITERATION AND AVERAGE OUT
	# THE ERROR AGAINST THE AVERAGED WEIGHT
	eout = 0.

	print '\nAPPROXIMATING OUT OF SAMPLE ERROR WITH 1000 POINTS'
	for i in range(iterations):
		if i % 20 == 0:
			print 'Iteration ' + str(i)
		dfout = getData(1000, m, b)
		eout += checkPoints(dfout, wav)

	print eout / iterations

def checkPointsPLA(df, w):
	bad_index = []
	for i in range(df.shape[0]):
		if (np.sign(np.dot(w, df.drop('side', axis = 1).iloc[i])) != \
					df.iloc[i]['side']):
			bad_index.append(i)
	if len(bad_index) == 0:
		return -1
	choose_rand = np.random.randint(0, len(bad_index))
	return bad_index[choose_rand]

def runPLA(df, m, y0):
	w = manualVersion(df)
	index = checkPointsPLA(df, w)
	iter_count = 0
	while index != -1:
		iter_count += 1
		row = df.iloc[index]
		w = w + row['side'] * row[['const', 'x', 'y']]
		index = checkPointsPLA(df, w)

	return w, iter_count

def plalinreg():
	iterations = 1000
	iterav = 0.

	for i in range(iterations):
		if (i % 20 == 0):
			print 'Iteration ' + str(i)
		m, b = pickLine()
		df = getData(10, m, b)
		w, iter_count = runPLA(df, m, b)
		iterav += iter_count

	iterav /= iterations
	plotData(df, m, b, w, 'plalinreg.png')

	print iterav

# ------------- NON-LINEAR TRANSFORMATION SECTION ----------

def getCircData(size, radsq):
	# CREATE DATA FOR POINTS IN X-SPACE
	df = pd.DataFrame((np.random.random(2 * size) * 2 - 1)\
			.reshape(size, 2), columns = ['x', 'y'])
	df['const'] = pd.Series(np.ones(size))
	df['side'] = df.apply(lambda coord : inCircle(coord, radsq), axis = 1)
	return df[['const', 'x', 'y', 'side']]

def inCircle(coord, radsq):
	return int(np.sign(coord['x']**2 + coord['y']**2 - radsq))

def addNoise(df, fraction):
	iterations = int(df.shape[0] * fraction)
	index_lst = []
	i = 0
	while i < iterations:
		if i not in index_lst:
			index = np.random.randint(df.shape[0])
			df['side'].iloc[index] *= -1
			index_lst.append(i)
			i += 1
	return df

def plotCircData(df, radsq, w, filename):
	# CREATE BOUNDARY AND ARRAY FOR REAL WEIGHTS
	plotlo = -1.2
	plothi = 1.2

	# CREATE ARRAYS FOR PLOTTING A CIRCLE
	x = np.linspace(-1, 1, 1000)
	y = np.linspace(-1, 1, 1000)
	xx, yy = np.meshgrid(x, y)
	F = xx**2 + yy**2 - radsq

	# SCATTER PLOT OF THE DATA AND REAL WEIGHTS LINE
	df.plot.scatter(x = 'x', y = 'y', c = 'side', cmap = 'coolwarm',
						colorbar = False)
	plt.xlim([plotlo, plothi])
	plt.ylim([plotlo, plothi])
	plt.contour(xx, yy, F, [0], colors = 'green')

	# PLOT THE WEIGHTS AND FILL THE ERROR REGION
	if w is not None:
		G = w[0] + w[1] * xx + w[2] * yy + w[3] * xx * yy + w[4] * xx**2 + \
			w[5] * yy**2
		plt.contour(xx, yy, G, [0], colors = ['black'])
		custom_lines = [Line2D([0], [0], color='green', lw=2),
						Line2D([0], [0], color='black', lw=2)]
		plt.legend(custom_lines, ['True Divider', 'Regression Divider'],
							loc='center left', bbox_to_anchor=(1, 0.5))

	if filename is not None:
		plt.savefig(filename, bbox_inches = 'tight', dpi = 900)
	plt.close()

def findCircleLinearFitError(iterations):
	err = 0.
	for i in range(iterations):
		print 'Iteration: ' + str(i)
		df = addNoise(getCircData(1000, 0.6), 0.1)
		w = manualVersion(df)
		if i == 0:
			plotCircData(df, 0.6, None, 'nonLinearTransform.png')
			plotData(df, None, None, w, 'nonLinearLineFit.png')
		err += checkPoints(df, w)
	print '\nAverage LinReg Error: ' + str(err / iterations)

def LinToQuadTransform(df):
	dfquad = df.copy()
	dfquad['xy'] = df['x'] * df['y']
	dfquad['x2'] = df['x']**2
	dfquad['y2'] = df['y']**2
	return dfquad[['const', 'x', 'y', 'xy', 'x2', 'y2', 'side']]

def nonLinearTransformProblem():
	iterations = 100
	#findCircleLinearFitError(10)

	df = addNoise(getCircData(1000, 0.6), 0.1)
	dfquad = LinToQuadTransform(df)
	wquad = np.zeros(len(dfquad.columns) - 1)
	print '\nGETTING NONLINEAR WEIGHTS'
	for i in range(iterations):
		wquad += manualVersion(dfquad)
	wquad /= iterations

	print '\nESTIMATING EOUT'
	eout = 0.
	for i in range(iterations):
		print 'Iteration: ' + str(i)
		newdf = addNoise(getCircData(1000, 0.6), 0.1)
		newdfquad = LinToQuadTransform(newdf)
		eout += checkPoints(newdfquad, wquad)
		if i == 0:
			plotCircData(df, 0.6, wquad, 'nonlinregerrplot.png')
	print 'Average Eout Error: ' + str(eout / iterations)


if __name__=='__main__':
	#coin_analysis(1000, 10, 100000)
	#RegExercise()
	#plalinreg()
	nonLinearTransformProblem()