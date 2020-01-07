import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd
import warnings
plt.rcParams["patch.force_edgecolor"] = True
warnings.filterwarnings('ignore')

def getData(size):
	df = pd.DataFrame((np.random.random(2 * size) * 2 - 1)\
			.reshape(size, 2), columns = ['x', 'y'])
	df['const'] = pd.Series(np.ones(size))
	return df[['const', 'x', 'y']]

def pickLine():
	p = np.random.random(4) * 2 - 1

	m = (p[3] - p[2]) / (p[1] - p[0])

	y0 = p[2] - m * p[0]

	return m, y0

def pickSide(coord, m, y0):
	return int(np.sign(coord['y'] - m * coord['x'] - y0))

def plotData(df, m, y0, w, filename):
	plotlo = -1.2
	plothi = 1.2
	xx = np.linspace(plotlo, plothi, 1000)
	yy = m * xx + y0

	df.plot.scatter(x = 'x', y = 'y', c = 'side', cmap = 'coolwarm',
						colorbar = False)
	plt.xlim([plotlo, plothi])
	plt.ylim([plotlo, plothi])
	plt.plot(xx, yy, 'k', label='True Divider')
	if w is not None:
		ypla = - (w[0] + w[1] * xx) / w[2]
		plt.plot(xx, ypla, label='PLA Divider')
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ypla = - (w[0] + w[1] * xx) / w[2]
	plt.fill_between(xx, yy, ypla, color='grey', alpha=0.5)

	if filename is not None:
		plt.savefig(filename, bbox_inches = 'tight', dpi = 900)
	plt.close()

def checkPointsPLA(df, w):
	for i in range(df.shape[0]):
		if (np.sign(np.dot(w, df.drop('side', axis = 1).iloc[i])) != \
					df.iloc[i]['side']):
			return i
	return -1

def linef(x, m1, m2, b1, b2):
	top = max(m1 * x + b1, m2 * x + b2)
	bot = min(m1 * x + b1, m2 * x + b2)
	top = min(top, 1.)
	bot = max(bot, -1.)
	if top < -1 or bot > 1:
		return 0
	return top - bot


def calcErrProb(w, m1, b1):
	m2 = - w[1] / w[2]
	b2 = - w[0] / w[2]

	return quad(linef, -1., 1., args = (m1, m2, b1, b2))[0] / 4.



def runPLA(size, save_bool):
	if save_bool:
		filename = 'ProblemSet1_pla2.png'
	else:
		filename = None

	df = getData(size)
	m, y0 = pickLine()

	# COLOR THE POINTS BASED ON WHICH SIDE OF THE LINE THE POINT IS ON
	df['side'] = df.apply(lambda coord : pickSide(coord, m, y0), axis = 1)

	w = np.array([0., 0., 0.]) 
	index = checkPoints(df, w)
	iter_count = 0
	while index != -1:
		iter_count += 1
		row = df.iloc[index]
		w = w + row['side'] * row[['const', 'x', 'y']]
		index = checkPoints(df, w)
	plotData(df, m, y0, w, filename)

	err_prob = calcErrProb(w, m, y0)

	return iter_count, err_prob
		
# MAIN PLA PROBLEM SET
def PLAProblem():
	size = 100

	# PLOT
	sns.set_style('darkgrid')

	PLA_table = pd.DataFrame()
	iter_lst = []
	errorprob_lst = []
	iterations = 1000

	for i in range(iterations):
		print 'iteration ' + str(i)
		iter_count, err_prob = runPLA(size, i == iterations - 1)
		iter_lst.append(iter_count)
		errorprob_lst.append(err_prob)

	c1 = 'Iterations'
	c2 = 'Error Probability'
	PLA_table[c1] = pd.Series(iter_lst)
	PLA_table[c2] = pd.Series(errorprob_lst)
	print 'Average Iteration Length: ' + str(PLA_table[c1].mean())
	print 'Iteration Length Deviation: ' + str(PLA_table[c1].std())
	print 'Average Error Probability: ' + str(PLA_table[c2].mean())
	print 'Error Probability Deviation: ' + str(PLA_table[c2].std())

	sns.distplot(PLA_table[c1], kde = False)
	plt.xlim(xmin = 0)
	plt.savefig('iterationshist2.png', bbox_inches = 'tight')
	plt.close()

	sns.distplot(PLA_table[c2], kde = False, bins = 50)
	plt.xlim(xmin = 0)
	plt.savefig('errorprob2.png', bbox_inches = 'tight')
	plt.close()


# ORGANIZE CODE BY SECTION
if __name__=='__main__':
	#PLAProblem()
