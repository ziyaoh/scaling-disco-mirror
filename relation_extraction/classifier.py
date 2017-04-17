from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class Classifier:
	"""
	A classifier extracts feature vectors from data, and fits linear classifier to the passed in data.
	"""

	def fit(self, X, y):
		"""
		Fit the transform and transform the data, then fit the transformed data using the final estimator.

		:param X: Training data. An iterable which yields string of sentences.

		:param y: Target vector relative to X.
		"""
		raise NotImplementedError

	def predict(self, X):
		"""
		Predict class labels for samples in X.

		:param X: Samples.

		:rtype: List of predicted class label per sample.
		"""
		raise NotImplementedError


class LinearClassifier(Classifier):

	def __init__(self, feature, classifier):
		if feature == 'default':
			self.feature = CountVectorizer()
		if classifier == 'logit':
			self.classifier = LogisticRegression()
		self.pipe = Pipeline(steps=[('feature', self.feature), ('classifier', self.classifier)])

	def fit(self, X, y):
		self.pipe.fit(X, y)

	def predict(self, X):
		return self.pipe.predict(X)