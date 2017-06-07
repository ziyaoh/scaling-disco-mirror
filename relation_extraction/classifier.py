from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.linear_model import LogisticRegression


class Classifier:
    """
    A classifier object extracts feature vectors from data, and fits the specified classifier to the passed in data.
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
    def __init__(self, feature='unigram', classifier='logit'):
        print feature, classifier

        if feature == 'unigram':
            self.vectorizer = CountVectorizer()
        elif feature == 'semantic':
            self.vectorizer = MultiLabelBinarizer()

        if classifier == 'logit':
            self.classifier = LogisticRegression()

        self.pipe = Pipeline(steps=[('feature', self.vectorizer), ('classifier', self.classifier)])

    def fit(self, X, y):
        # self.pipe.fit(X, y)
        self.vectorizer.fit(X)
        features = self.vectorizer.transform(X)
        print 'length vocabulary:', len(features[0])
        self.classifier.fit(features, y)

    def predict(self, X):
        features = self.vectorizer.transform(X)
        return self.classifier.predict(features)