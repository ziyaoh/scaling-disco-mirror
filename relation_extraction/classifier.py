import sys
from modelTest import get_confusion_table, get_accuracy, get_precision_recall, get_f1_score
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

    def evaluate(self, X, y, report):
        raise NotImplementedError

    def get_relations(self):
        return self.relations


class LinearClassifier(Classifier):
    def __init__(self, feature='unigram', classifier='logit'):
        print feature, classifier

        self.relations = []

        if feature == 'unigram':
            self.vectorizer_x = CountVectorizer(binary=True)
        elif feature == 'semantic':
            self.vectorizer_x = CountVectorizer(analyzer=lambda x: x, binary=True)
        self.vectorizer_y = CountVectorizer()

        if classifier == 'logit':
            self.classifier = LogisticRegression()

        # self.pipe = Pipeline(steps=[('feature', self.vectorizer), ('classifier', self.classifier)])

    def fit(self, X, y):
        # self.pipe.fit(X, y)
        self.relations = list(set(y))

        self.vectorizer_x.fit(X)
        features = self.vectorizer_x.transform(X)
        y_binary = self.vectorizer_y.fit_transform(y)
        print 'length vocabulary:', len(self.vectorizer_x.vocabulary_)
        self.classifier.fit(features, y)

    def predict(self, X):
        features = self.vectorizer_x.transform(X)
        pred = self.classifier.predict(features)
        print pred
        return pred

    def evaluate(self, X_test, y_test, report):
        y_pred = self.predict(X_test)

        confusion_table = get_confusion_table(y_pred, y_test, self.relations)
        accuracy = get_accuracy(y_pred, y_test)
        precision_recall = get_precision_recall(confusion_table)
        f1_micro = get_f1_score(y_pred, y_test, 'micro')
        f1_macro = get_f1_score(y_pred, y_test, 'macro')

        self.generate_report(report, confusion_table, accuracy, precision_recall, f1_micro, f1_macro, self.relations)

    def generate_report(self, output_file, confusion_table, accuracy, precision_recall, f1_micro, f1_macro, original_relations):
        """
        Generate a report based on testing data.
        """
        relations = [rel.split('/')[-1] for rel in original_relations]
        with open(output_file, 'w') as writer:
            writer.write("\nAll classes: ")
            for relation in relations:
                writer.write("%s " % relation)
            writer.write("\n")

            writer.write("\nPrediction Accuracy: %s\n" % accuracy)

            writer.write("\nConfusion Table:\n")
            width = 20
            header = "".rjust(width)
            for column in relations:
                header += column.rjust(width)
            writer.write(header)
            for actual_relation in relations:
                writer.write("\n%s" % actual_relation.rjust(width))
                for pred_relation in relations:
                    num = confusion_table[actual_relation][pred_relation] \
                        if actual_relation in confusion_table and pred_relation in confusion_table[actual_relation] else 0
                    writer.write("%s" % str(num).rjust(width))
            writer.write("\n")

            writer.write("\nF1 Scores:\n")
            writer.write("micro: %s\n" % f1_micro)
            writer.write("macro: %s\n" % f1_macro)
            writer.write("\n")

            writer.write("\nPrecision and Recall:\n")
            writer.write("".rjust(width) + "precision".rjust(10) + "recall".rjust(10))

            for i, relation in enumerate(relations):
                p_r = precision_recall[original_relations[i]]
                writer.write("\n" + relation.rjust(width) + "%.4f".rjust(10) % p_r[0] + "%.4f".rjust(10) % p_r[1])
            writer.write("\n")


class OneVsRestClassifier(Classifier):
    def __init__(self, feature='unigram', classifier='logit'):
        print feature, classifier

        self.relations = []

        if feature == 'unigram':
            self.vectorizer_x = CountVectorizer()
            self.vectorizer_y = CountVectorizer()
        elif feature == 'semantic':
            self.vectorizer_x = CountVectorizer(analyzer=lambda x: x)
            self.vectorizer_y = CountVectorizer(analyzer=lambda x: x)

        if classifier == 'logit':
            self.classifier = OVRClassifier(LogisticRegression(class_weight='balanced'))

    def fit(self, X, y):
        relations_set = set()
        for ins_labels in y:
            relations_set = relations_set.union(set(ins_labels))
        self.relations = list(relations_set)

        X_vec = self.vectorizer_x.fit_transform(X)
        y_vec = self.vectorizer_y.fit_transform(y)

        y_for_test = self.classify_by_label(y_vec.toarray())
        for label in y_for_test:
            print label, len(y_for_test[label]), sum(y_for_test[label]), len(y_for_test[label]) - sum(y_for_test[label])

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

    def evaluate(self, X_test, y_test, report):
        
        y_pred_binary = self.predict(X_test, binarized=True)
        y_test_binary = self.binarize_label(y_test)

        #print y_pred_binary['per:origin'], len(y_pred_binary['per:origin'])
        #print y_test_binary['per:origin'], len(y_test_binary['per:origin'])

        f1s = {}
        overall_f1 = 0.0
        for label in y_pred_binary:
            f1s[label] = (get_f1_score(y_pred_binary[label], y_test_binary[label], 'macro'), get_confusion_table(y_pred_binary[label], y_test_binary[label], (0, 1)))
            overall_f1 += f1s[label][0]
        overall_f1 /= len(y_pred_binary)

        with open(report, 'w') as writer:
            for label in f1s:
                writer.write("%s\t%s\n" % (label, f1s[label][0]))
                confusion_table = f1s[label][1]
                writer.write("\t%s\t%s\n" % (1, 0))
                writer.write("1\t%s\t%s\n" % (confusion_table[1][1], confusion_table[1][0]))
                writer.write("0\t%s\t%s\n" % (confusion_table[0][1], confusion_table[0][0]))
            writer.write("overall F1: %s\n" % overall_f1)
            writer.write("\n")

def construct_classifier(feature, classifier_type):
    if classifier_type == 'linear':
        return LinearClassifier(feature)
    elif classifier_type == 'OneVsRest':
        return OneVsRestClassifier(feature)
    else:
        print 'unknown classifier type: %s' % classifier_type
        sys.exit()
