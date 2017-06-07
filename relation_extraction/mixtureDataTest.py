import os
import sys
import classifier
from dataParse import construct_dataReader
from modelTest import draw_f1_curve, model_test
from relationExtraction import generate_report
from sklearn.model_selection import train_test_split


base_file = 'train_DS'
input_9000 = 'DS_random_9000.txt'
input_18000 = 'DS_random_18000.txt'

CV_proportion = 0.3


def create_mixture_data():
    os.system("python dataCooperation.py %s Naacl -i train_CS_random -f Naacl -p 9000 -o %s" % (base_file, input_9000))
    os.system("python dataCooperation.py %s Naacl -i train_CS_random -f Naacl -p 18000 -o %s" % (base_file, input_18000))


def get_data(test_file, feature, cross_validation=False):
    if feature != 'unigram' and feature != 'semantic':
        print 'unknown feature type'
        sys.exit()

    print 'reading data'
    parser_base = construct_dataReader(base_file, 'Naacl')
    parser_9000 = construct_dataReader(input_9000, 'standard')
    parser_18000 = construct_dataReader(input_18000, 'standard')

    X_base, y_base = parser_base.read_format_data(feature)
    X_9000, y_9000 = parser_9000.read_format_data(feature)
    X_18000, y_18000 = parser_18000.read_format_data(feature)

    if cross_validation:
        X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(X_base, y_base, test_size=CV_proportion)
        X_9000_train, X_9000_test, y_9000_train, y_9000_test = train_test_split(X_9000, y_9000, test_size=CV_proportion)
        X_18000_train, X_18000_test, y_18000_train, y_18000_test = train_test_split(X_18000, y_18000, test_size=CV_proportion)

        datasets = {
            base_file: (X_base_train, X_base_test, y_base_train, y_base_test, parser_base.relations),
            input_9000: (X_9000_train, X_9000_test, y_9000_train, y_9000_test, parser_9000.relations),
            input_18000: (X_18000_train, X_18000_test, y_18000_train, y_18000_test, parser_18000.relations)
        }
    else:
        parser_test = construct_dataReader(test_file, 'Naacl')
        X_test, y_test = parser_test.read_format_data(feature)

        datasets = {
            base_file: (X_base, X_test, y_base, y_test, parser_base.relations),
            input_9000: (X_9000, X_test, y_9000, y_test, parser_9000.relations),
            input_18000: (X_18000, X_test, y_18000, y_test, parser_18000.relations)
        }

    return datasets


def train_models(datasets, feature):
    classifier = build_model(base_file, datasets[base_file][0], datasets[base_file][2], feature)
    classifier_9000 = build_model(input_9000, datasets[input_9000][0], datasets[input_9000][2], feature)
    classifier_18000 = build_model(input_18000, datasets[input_18000][0], datasets[input_18000][2], feature)

    return (classifier, classifier_9000, classifier_18000)


def build_model(input_file, X, y, feature):
    """
    Read training data from input file, fit a classifier model according to the training data.
    Read testing data from test file, and test the classifier model.
    """
    print "training on %s" % input_file

    my_classifier = classifier.LinearClassifier(feature=feature)
    my_classifier.fit(X, y)
    return my_classifier


def test_models(classifiers, datasets):
    """
    test_sets = {
        0: (classifier_base, X_base_test, y_base_test),
        ...
    }
    """

    print "testing classifiers"

    test_sets = {
        0: (classifiers[0], datasets[base_file][1], datasets[base_file][3]),
        9000: (classifiers[1], datasets[input_9000][1], datasets[input_9000][3]),
        18000: (classifiers[2], datasets[input_18000][1], datasets[input_18000][3])
    }

    draw_f1_curve(test_sets)

    # (confusion_table, accuracy, precision_recall, f1_micro, f1_macro) = model_test(classifiers[1], test_sets['9000'][1],
    #                                                                                          test_sets['9000'][2], datasets[input_9000][4])
    # generate_report('unigram', 'logit', '9000_report.txt', confusion_table, accuracy, precision_recall, f1_micro, f1_macro)


if __name__ == '__main__':
    create_input = False 
    if create_input:
        create_mixture_data()
    else:
        feature = 'semantic'
        datasets = get_data('train_CS_gabor', feature, cross_validation=False)
        classifier, classifier_9000, classifier_18000 = train_models(datasets, feature)
        test_models([classifier, classifier_9000, classifier_18000], datasets)
