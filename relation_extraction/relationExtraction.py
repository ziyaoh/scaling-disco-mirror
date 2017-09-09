import argparse
import sys
import pickle
import classifier
import modelTest
from contextlib import contextmanager
from dataParse import construct_dataReader
from classifier import construct_classifier


def read_command():
    """
    input: input file name
    test: test file name
    -c: classifier type
    -f: feature type
    -o: output file name
    -d: input data file format
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the input file name')
    parser.add_argument('test', help='the held out test data')
    parser.add_argument('-f', '--feature',
                        default='unigram',
                        help='The type of feature we extract from the input data. If not specified, unigram features '
                             'will be used.')
    parser.add_argument('-c', '--classifier',
                        default='linear',
                        choices=['linear', 'OneVsRest'],
                        help='The type of classifier we want to use. If not specified, logistic regression classifier '
                             'will be used.')
    parser.add_argument('-o', '--output',
                        default=None,
                        help='The testing report file name. If not specified, report will be printed to standard out.')
    parser.add_argument('-d', '--dataformat',
                        default='standard',
                        choices=['SemEval', 'Naacl', 'standard', 'DefaultFeaturized'],
                        help='The input file data format. We will parse input file according to this information. '
                             'If not specified, SemEval format will be used.')

    return parser.parse_args()


def parse_data(input_file, data_format, feature_type, classifier_type):
    parser = construct_dataReader(input_file, data_format)

    print 'parsing'
    (X, y_list) = parser.read_format_data(feature_type)

    y = []
    if classifier_type == 'linear':
        for labels in y_list:
            if len(labels) == 0:
                y.append('NA')
            else:
                y.append(labels[0])
    elif classifier_type == 'OneVsRest':
        y = y_list

    signiture = input_file.split("/")[-1].split(".")[0]
    return X, y, parser.relations, signiture


def build_model(X, y, feature_type, classifier_type, signiture):
    """
    Read training data from input file, fit a classifier model according to the training data.
    Read testing data from test file, and test the classifier model.
    """

    my_classifier = construct_classifier(feature_type, classifier_type, signiture)
    my_classifier.fit(X, y)
    return my_classifier


def test_model(my_classifier, X_test, y_test, report=None):
    """
    Test the classifier on testing data, and return resulting confusion table, prediction accuracy, precision_recall and
    all relations in testing data.
    """

    #(confusion_table, accuracy, precision_recall, f1_micro, f1_macro) = modelTest.model_test(my_classifier, X_test, y_test, relations)
    #return (confusion_table, accuracy, precision_recall, f1_micro, f1_macro)
    if report is None:
        report = '%s.score' % my_classifier.get_signiture()
    my_classifier.evaluate(X_test, y_test, report)


def save_model(classifier):
    model_file = '%s.model' % classifier.get_signiture()
    pickle.dump(classifier, open(model_file, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_model(model_file):
    return pickle.load(open(model_file, 'rb'))


if __name__ == '__main__':
    opt = read_command()
    X, y, relations, signiture = parse_data(opt.input, opt.dataformat, opt.feature, opt.classifier)
    my_classifier = build_model(X, y, opt.feature, opt.classifier, signiture)
    X_test, y_test, relations_test, signiture_test = parse_data(opt.test, opt.dataformat, opt.feature, opt.classifier)
    test_model(my_classifier, X_test, y_test)
