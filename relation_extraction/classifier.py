from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier as OVRClassifier


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
            self.vectorizer = CountVectorizer(analyzer=lambda x: x)

        if classifier == 'logit':
            self.classifier = LogisticRegression()

        self.pipe = Pipeline(steps=[('feature', self.vectorizer), ('classifier', self.classifier)])

    def fit(self, X, y):
        # self.pipe.fit(X, y)
        self.vectorizer.fit(X)
        features = self.vectorizer.transform(X)
        print 'length vocabulary:', len(self.vectorizer.vocabulary_)
        self.classifier.fit(features, y)

    def predict(self, X):
        features = self.vectorizer.transform(X)
        return self.classifier.predict(features)


class OneVsRestClassifier(Classifier):
    def __init__(self, feature='unigram', classifier='logit'):
        print feature, classifier

        if feature == 'unigram':
            self.vectorizer_x = CountVectorizer()
            self.vectorizer_y = CountVectorizer()
        elif feature == 'semantic':
            self.vectorizer_x = CountVectorizer(analyzer=lambda x: x)
            self.vectorizer_y = CountVectorizer(analyzer=lambda x: x)

        if classifier == 'logit':
            self.classifier = OVRClassifier(LogisticRegression())

    def fit(self, X, y):
        X_vec = self.vectorizer_x.fit_transform(X)
        y_vec = self.vectorizer_y.fit_transform(y)

        self.classifier.fit(X_vec, y_vec)

    def predict(self, X, binarized=False):
        X_test_vec = self.vectorizer_x.transform(X)
        y_pred_vec = self.classifier.predict(X_test_vec)

        if binarized:
            y_vec = y_pred_vec.toarray()
            return self.classify_by_label(y_vec)
        else:
            return self.vectorizer_y.inverse_transform(y_pred_vec)

    def binarize_label(self, y):
        y_binary = self.vectorizer_y.transform(y).toarray()
        return self.classify_by_label(y_binary)

    def classify_by_label(self, instances):
        pred = {}
        for label in self.vectorizer_y.vocabulary_:
            pred[label] = []
            index = self.vectorizer_y.vocabulary_[label]
            for ins in instances:
                pred[label].append(ins[index])
        return pred
