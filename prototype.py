import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

def transform(x):
	val = -1
	if abs(x) <= 0.05:
		val = 0 
	elif x > 0:
		val = 1
	return val

def load_data(filename):
	ratings = pd.read_csv(filename, encoding='latin1')
	df = pd.DataFrame(ratings)
	df = df.dropna()
	# make a nnew column as dataframe["votes"] with ratio of candidatevotes/totalvotes
	df['FIPS'] = df['FIPS'].astype(int)
	df['votes'] = df['candidatevotes']/df['totalvotes']
	df['votes'] = df['votes']-0.5
	df['votes'] = df['votes'].apply(lambda x: transform(x))

	# df['votes'] = df['votes'] + abs(df['votes'])
	# df['votes'] =(df['votes']-df['votes'].min())/(df['votes'].max()-df['votes'].min())
	print(df['votes'])
	reader = Reader(rating_scale=(0, 1))
	data = Dataset.load_from_df(df[['FIPS', 'candidate', 'votes']], reader)
	ts = data.build_full_trainset()
	counties = df['FIPS'].values
	return ts, counties


def trainSVD(training_data, colorlabels, plot=True, savefig=True): #colorlabels, sizelabels, plot=True, savefig=True
	algo = SVD(n_factors=10, n_epochs=500, biased=False)
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

	# U = U.transpose()
	# A = np.linalg.svd(U)[0]
	# U_proj = np.dot(A[:, :2].transpose(), U)
	# # Rescale dimensions
	# U_proj /= U_proj.std(axis=1).reshape(2, 1)

	# if plot:
	# 	fig = plt.figure(figsize = (8,8))
	# 	ax = fig.add_subplot(1,1,1) 
	# 	ax.set_xlabel('First', fontsize = 15)
	# 	ax.set_ylabel('Second', fontsize = 15)
	# 	ax.set_title('Reduced SVD', fontsize = 20)
	# 	scatter = ax.scatter(U[0], U[1], c=colorlabels, s=10)
	# 	ax.grid()
	# 	cbar = fig.colorbar(scatter, ax=ax)
	# 	cbar.set_label("state")
	# 	if savefig:
	# 		plt.savefig("figures/svd_counties")
	# 	plt.show()

def visualize(filename):
	data, counties = load_data(filename)
	counties = list(dict.fromkeys(counties))
	for index, county in enumerate(counties):
		modified = str(county)
		if len(modified) == 4:
			modified = '0'+modified
		counties[index] = modified
	states_labels = [int(fips[0:2]) for fips in counties]
	trainSVD(data, states_labels)

if __name__ == '__main__':
	filename = 'data/countypres_2000-2016.csv'
	visualize(filename)

# from surprise import Dataset
# from surprise import accuracy
# from surprise import SVD, evaluate
# from surprise import NMF
# from surprise import KNNBasic
# from surprise import Reader
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# movie_features = ["Movie Id", "Movie Title", "Unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime",
# 		 "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
# 		 "Thriller", "War", "Western"]
# movies = pd.read_csv('../data/movies.txt', delimiter='\t', header=None, encoding='latin1', names=movie_features)

# def load_data(filename = '../data/data.txt'):
# 	rating_features = ["User Id","Movie Id","Rating"]
# 	ratings = pd.read_csv('../data/data.txt', delimiter='\t', header=None, encoding='latin1', names=rating_features)
# 	df = pd.DataFrame(ratings)
# 	reader = Reader(rating_scale=(0.5, 5.0))
# 	data = Dataset.load_from_df(df[['User Id', 'Movie Id', 'Rating']], reader)
# 	ts = data.build_full_trainset()
# 	return ts

# def trainSVD(make_plot=True):
# 	ts = load_data()
# 	algo = SVD(n_factors=25, n_epochs=300, biased=False)
# 	# algo = NMF(n_factors=2, n_epochs=100)
# 	# algo = KNNBasic()

# 	algo.fit(ts)
# 	pred = algo.test(ts.build_testset())
# 	accuracy.rmse(pred)
# 	algo.qi.shape
# 	V = algo.qi
# 	V = V.transpose()
# 	# Project to 2 dimensions
# 	A = np.linalg.svd(V)[0]
# 	V_proj = np.dot(A[:, :2].transpose(), V)
# 	# Rescale dimensions
# 	V_proj /= V_proj.std(axis=1).reshape(2, 1)
# 	if(make_plot):
# 		plot(V, ts)

# 	return V_proj

# def plot(V, ts):
# 	scores = np.array(list(map(lambda x: np.average([v for _,v in x]), ts.ir.values())))
# 	num_scores = np.array(list(map(lambda x: len(x), ts.ir.values())))
# 	years = np.array(movies['Movie Title'].str.extract('\((\d*)\)').values)
# 	titles = np.array(movies['Movie Title'].values)


# 	outlier_idx = [int(ts.to_raw_iid(i))-1 for i in np.where(np.linalg.norm(V, axis=1) > 1000)[0]]
# 	popular_idx = [int(ts.to_raw_iid(i))-1 for i in np.where(num_scores > 350)[0]]
# 	interesting_movies = movies[['Movie Id', 'Movie Title']].values[np.unique(outlier_idx + popular_idx)]

# 	# print(interesting_movies)

# 	plt.figure(figsize=(8,8))
# 	plt.scatter(*V.T, c=scores, s=num_scores, cmap=plt.get_cmap('RdYlGn'), alpha=0.6)
# 	plt.colorbar().set_label("Average ratings")
# 	plt.title("Unbiased SVD Surprise!")

# 	texts = []
# 	for movie_id, name in interesting_movies:
# 		print(movie_id)
# 		iid = movie_id
# 		texts.append(plt.annotate(name, xy=V[iid], horizontalalignment='center', textcoords='offset points', 
# 								  xytext=(0, 0), arrowprops=dict(arrowstyle='-', lw=1, alpha=0.5)))

# 	plt.show()


# if __name__ == '__main__':
# 	trainSVD()



# # Load the movielens-100k dataset (download it if needed).
# data = Dataset.load_builtin('ml-100k')

# # Use the famous SVD algorithm.
# algo = SVD()

# # Run 5-fold cross-validation and print results.
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# ########################################################################


# # Use movielens-100K
# data = Dataset.load_builtin('ml-100k')

# param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
#               'reg_all': [0.4, 0.6]}
# gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

# gs.fit(data)

# # best RMSE score
# print(gs.best_score['rmse'])

# # combination of parameters that gave the best RMSE score
# print(gs.best_params['rmse'])

# # We can now use the algorithm that yields the best rmse:
# algo = gs.best_estimator['rmse']
# algo.fit(data.build_full_trainset())

# import pandas as pd  # noqa
# results_df = pd.DataFrame.from_dict(gs.cv_results)



