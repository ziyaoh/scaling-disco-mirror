import argparse
import dataParse
import classifier
import modelTest
from dataParse import SemEvalReader


def read_command():
    """
    input: input file name
    test: test file name
    -c: classifier type
    -f: feature type
    -o: output file name
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the input file name')
    parser.add_argument('test', help='the held out test data')
    parser.add_argument('-f', '--feature',
                        default='default',
                        help='The type of feature we extract from the input data. If not specified, uniary feature will be used.')
    parser.add_argument('-c', '--classifier',
                        default='logit',
                        choices=['logit', "NN"],
                        help='The type of classifier we want to use. If not specified, logistic regression classifier will be used.')
    parser.add_argument('-o', '--output',
                        default=None,
                        help='The testing report file name. If not specified, report will be printed to standard out.')

    return parser.parse_args()


def build_model(input_file):
    """
    Read training data from input file, fit a classifier model according to the training data.
    Read testing data from test file, and test the classifier model.
    """
    parser = SemEvalReader(input_file)
    (X, y) = parser.read_format_data()

    my_classifier = classifier.LinearClassifier(opt.feature, opt.classifier)
    my_classifier.fit(X, y)
    return my_classifier


def test_model(my_classifier, test_file, output_file):
    parser = SemEvalReader(test_file)
    (X_test, y_test) = parser.read_format_data()

    (confusion_table, precision_recall) = modelTest.model_test(my_classifier, X_test, y_test)
    # confusion_table = modelTest.model_test(my_classifier, X_test, y_test)
    for key in confusion_table:
        print key

        print precision_recall[key][0], precision_recall[key][1]
        # print confusion_table[key]


if __name__ == '__main__':
    opt = read_command()
    my_classifier = build_model(opt.input)
    test_model(my_classifier, opt.test, opt.output)
