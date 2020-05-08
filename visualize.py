# Visualize how similar counties are by plotting the first two columns (abstract features) of the left matrix
# Visualize how similar candidates are by plotting the first two columns (abstract features) of the right matrix








# from surprise import Dataset
# from surprise import Reader
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import svd_surprise as method3

# class visualization:
# 	movie_features = ["Movie Id", "Movie Title", "Unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime",
#          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
#          "Thriller", "War", "Western"]
# 	movies = pd.read_csv('../data/movies.txt', delimiter='\t', header=None, encoding='latin1', names=movie_features)

# 	def __init__(self, V, source):
# 		rating_features = ["User Id","Movie Id","Rating"]
# 		ratings = pd.read_csv(source, delimiter='\t', header=None, encoding='latin1', names=rating_features)
# 		df = pd.DataFrame(ratings)
# 		reader = Reader(rating_scale=(0.5, 5.0))
# 		data = Dataset.load_from_df(df[['User Id', 'Movie Id', 'Rating']], reader)
# 		self.V = V
# 		self.ts = data.build_full_trainset()
# 		self.scores, self.num_scores = self.get_series()
# 		self.interesting_movies = self.get_interesting_movies()

# 	def get_series(self):
# 		data = self.ts
# 		# scores = np.array(list(map(lambda x: np.average([i for _,i in x]), data.ir.values())))
# 		scores = np.array(self.movies['Movie Title'].str.extract('\((\d*)\)').values)
# 		scores = scores.flatten()
# 		num_scores = np.array(list(map(lambda x: len(x), data.ir.values())))
# 		return scores, num_scores

# 	def get_interesting_movies(self):
# 		data = self.ts
# 		outlier_idx = [int(data.to_raw_iid(i))-1 for i in np.where(np.linalg.norm(self.V, axis=1) > 1000)[0]]
# 		popular_idx = [int(data.to_raw_iid(i))-1 for i in np.where(self.num_scores > 300)[0]]
# 		interesting_movies = self.movies[['Movie Id', 'Movie Title']].values[np.unique(popular_idx)]
# 		return interesting_movies

# 	def make_plot(self):
# 		plt.figure(figsize=(8,8))
# 		plt.scatter(*self.V, c=self.scores, s=self.num_scores, cmap=plt.get_cmap('RdYlGn'), alpha=0.6) #cmap: rainbow_r, Blues, RdYlGn
# 		plt.colorbar().set_label("Ratings")
# 		plt.title("Unbiased SVD Surprise!")

# 		texts = []
# 		for movie_id, name in self.interesting_movies:
# 			iid = movie_id
# 			texts.append(plt.annotate(name, xy=self.V.T[iid], horizontalalignment='center', textcoords='offset points', 
# 									  xytext=(0, 0), arrowprops=dict(arrowstyle='-', lw=1, alpha=0.5)))
# 		plt.savefig('visualization.png')
# 		plt.show()

# if __name__ == '__main__':
# 	V1 = method3.trainSVD(False)
# 	viz1 = visualization(V1, '../data/data.txt')
# 	viz1.make_plot()
