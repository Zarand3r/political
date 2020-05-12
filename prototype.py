import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

def transform(x):
	# print(x)
	x = x-0.5
	val = -1
	if abs(x) <= 0.05:
		val = 0 
	elif x > 0:
		val = 1
	# print(val)
	return val

def load_data(filename):
	ratings = pd.read_csv(filename, encoding='latin1')
	df = pd.DataFrame(ratings)
	df = df.dropna()
	# make a nnew column as dataframe["votes"] with ratio of candidatevotes/totalvotes
	df['FIPS'] = df['FIPS'].astype(int)
	df['candidate'] = df['year'].map(str) + "_" + df['candidate']
	df['votes'] = df['candidatevotes']/df['totalvotes']
	df['votes'] = df['votes'].apply(lambda x: transform(x))
	# df['votes'] = df['votes']-0.5

	# df['votes'] = df['votes'] + abs(df['votes'])
	# df['votes'] =(df['votes']-df['votes'].min())/(df['votes'].max()-df['votes'].min())
	return df[['FIPS', 'candidate', 'votes']]

def trainSVD(X, colorlabels, plot=True, savefig=True):
	U, S, V = np.linalg.svd(X, full_matrices=False)
	if plot:
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('First', fontsize=15)
		ax.set_ylabel('Second', fontsize=15)
		ax.set_title('Algebraic SVD', fontsize=20)
		scatter = ax.scatter(U[:,0], U[:,1], c=colorlabels, s=10, alpha=0.7) #explore labeling colors with features like demographics, age
		ax.grid()
		cbar = fig.colorbar(scatter, ax=ax)
		cbar.set_label("state")
		if savefig:
			plt.savefig("figures/algebraic_svd_counties")
		plt.show()


def trainSVD_surprise(training_data, colorlabels, plot=True, simplify=False, savefig=True): #colorlabels, sizelabels, plot=True, savefig=True
	# algo = SVD(n_factors=4, n_epochs=1000, biased=True)
	# algo = SVD(n_factors=20, n_epochs=500, biased=False)
	algo = SVDpp(n_factors=10, n_epochs=1000)
	algo.fit(training_data)
	U = algo.pu 
	if plot:
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('First', fontsize=15)
		ax.set_ylabel('Second', fontsize=15)
		ax.set_title('Reduced SVD', fontsize=20)
		scatter = ax.scatter(U[:,0], U[:,1], c=colorlabels, s=10, alpha=0.7) #explore labeling colors with features like demographics, age
		ax.grid()
		cbar = fig.colorbar(scatter, ax=ax)
		cbar.set_label("state")
		if savefig:
			plt.savefig("figures/svd_counties")
		plt.show()

	if simplify:
		U = U.transpose()
		A = np.linalg.svd(U)[0]
		U_proj = np.dot(A[:, :2].transpose(), U)
		# Rescale dimensions
		U_proj /= U_proj.std(axis=1).reshape(2, 1)
		if plot:
			fig = plt.figure(figsize = (8,8))
			ax = fig.add_subplot(1,1,1) 
			ax.set_xlabel('First', fontsize = 15)
			ax.set_ylabel('Second', fontsize = 15)
			ax.set_title('Reduced SVD', fontsize = 20)
			scatter = ax.scatter(U_proj[0], U_proj[1], c=colorlabels, s=10)
			ax.grid()
			cbar = fig.colorbar(scatter, ax=ax)
			cbar.set_label("state")
			if savefig:
				plt.savefig("figures/svd_counties_simplfied")
			plt.show()
		return U_proj

	return U


def trainSVD_surprise3D(training_data, colorlabels, plot=True, savefig=True): #colorlabels, sizelabels, plot=True, savefig=True
	# algo = SVD(n_factors=4, n_epochs=1000, biased=True)
	# algo = SVD(n_factors=20, n_epochs=500, biased=False)
	algo = SVDpp(n_factors=10, n_epochs=1000)
	algo.fit(training_data)
	U = algo.pu 
	if plot:
		fig = plt.figure(figsize = (8,8))
		# ax = fig.add_subplot(1,1,1) 
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlabel('First', fontsize=15)
		ax.set_ylabel('Second', fontsize=15)
		ax.set_title('Reduced SVD', fontsize=20)
		scatter = ax.scatter(U[:,0], U[:,1], U[:,2], c=colorlabels, s=10, alpha=0.7) #explore labeling colors with features like demographics, age
		ax.grid()
		cbar = fig.colorbar(scatter, ax=ax)
		cbar.set_label("state")
		if savefig:
			plt.savefig("figures/svd_counties")
		plt.show()

def visualize(filename):
	df = load_data(filename)
	reader = Reader(rating_scale=(0, 1))
	train_data = Dataset.load_from_df(df, reader)
	ts = train_data.build_full_trainset()
	counties = df['FIPS'].values
	counties = list(dict.fromkeys(counties))
	for index, county in enumerate(counties):
		modified = str(county)
		if len(modified) == 4:
			modified = '0'+modified
		counties[index] = modified
	states_labels = [int(fips[0:2]) for fips in counties]
	trainSVD_surprise(ts, states_labels)
	trainSVD_surprise3D(ts, states_labels)

	df = df.pivot_table('votes', 'FIPS', 'candidate')
	df.dropna(inplace=True)
	# df.dropna(inplace=True, axis=1, how="any")
	trainSVD(df.values, 'g')

if __name__ == '__main__':
	filename = 'data/countypres_2000-2016.csv'
	visualize(filename)


