import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class	CSP(BaseEstimator, TransformerMixin):
	def		__init__(self):
		pass

	def		save(self, file):
		np.save(file, self.filters)

	def		load(self, file):
		self.filters = np.load(file)

	def	transform(self, X):
		res = np.array([np.dot(self.filters.T, x) for x in X])
		res = (res**2).mean(axis=2)
		return (np.log(res))

	def fit(self, X, y=None):
		n_classes = len(np.unique(y))
		weights = []
		covs = []

		for i in range(n_classes):
			x = X[y == i]
			weights.append(len(x))
			tmp = np.cov(x[0])
			for j in range(1, weights[-1]):
				tmp += np.cov(x[j])
			covs.append(tmp / weights[-1])

		V = self.jade(covs)
		self.filters = V
		return (self)

	def		whiten(self, X):
		u, _, v = np.linalg.svd(X[0], full_matrices=False)
		W = np.dot(u, v)
		for i in range(len(X)):
			X[i] = np.dot(np.dot(W, X[i]), W.T)
		return (X)

	def		jade(self, X, eps=1e-12, n_iter_max=1000):
		X2 = np.concatenate(self.whiten(X), 0).T
		m, n = X2.shape
		V = np.eye(m)
		k = 0
		s = 1

		while ((np.abs(s) > eps) and (k < n_iter_max)):
			k += 1
			for p in range(m - 1):
				for q in range(p + 1, m):
					Ip = np.arange(p, n, m)
					Iq = np.arange(q, n, m)
					
					g = np.array([X2[p, Ip] - X2[q, Iq], X2[p, Iq] + X2[q, Ip]])
					g2 = np.dot(g, g.T)

					on_diag = g2[0, 0] - g2[1, 1]
					off_diag = g2[0, 1] + g2[1, 0]
					
					theta = 0.5 * np.arctan2(off_diag, on_diag + np.sqrt(on_diag**2 + off_diag**2))
					c = np.cos(theta)
					s = np.sin(theta)
					if (np.abs(s) > eps):
						tmp = X2[:, Ip].copy()
						X2[:, Ip] = c * X2[:, Ip] + s * X2[:, Iq]
						X2[:, Iq] = c * X2[:, Iq] - s * tmp

						tmp = X2[p, :].copy()
						X2[p, :] = c * X2[p, :] + s * X2[q, :]
						X2[q, :] = c * X2[q, :] - s * tmp

						tmp = V[:, p].copy()
						V[:, p] = c * V[:, p] + s * V[:, q]
						V[:, q] = c * V[:, q] - s * tmp
		return (V)

