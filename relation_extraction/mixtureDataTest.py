import os
import sys
import argparse
from dataParse import construct_dataReader
from classifier import LinearClassifier, OneVsRestClassifier
from modelTest import draw_f1_curve, model_test
from relationExtraction import generate_report
from sklearn.model_selection import train_test_split


base_file = 'train_DS'
input_9000 = 'DS_random_9000.txt'
input_18000 = 'DS_random_18000.txt'

CV_proportion = 0.3


def read_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='store_true', help='Create input file.')

    return parser.parse_args()


def create_mixture_data(sizes):
    running_env = 'unix'
    print 'generating input files'

    for size in sizes:
        os.system(get_command(size, env=running_env))
        os.system(get_command(size, 9000, env=running_env))
        os.system(get_command(size, 18000, env=running_env))


def get_command(base_size, additional_size=None, env='unix'):
    if env == 'windows':
        command = 'c:\Python27\python'
    elif env == 'unix':
        command = 'python'
    else:
        print "unknown running environment %s" % env
        sys.exit()

    if not additional_size:
        output_file = get_output_file(base_size)
        return "%s dataCooperation.py -i %s -f Naacl -p %s -o %s" % (command, base_file, base_size, output_file)
    else:
        output_file = get_output_file(base_size, additional_size)
        return "%s dataCooperation.py -i %s train_CS_random -f Naacl Naacl -p %s %s -o %s" % \
               (command, base_file, base_size, additional_size, output_file)


def get_output_file(base_size, additional_size=None):
    if not additional_size:
        output_file = "base_%s.txt" % base_size
    else:
        output_file = "base_%s_random_%s.txt" % (base_size, additional_size)
    return output_file


def get_data(base_size, test_file, feature, cross_validation=False):
    if feature != 'unigram' and feature != 'semantic':
        print 'unknown feature type'
        sys.exit()

    print 'reading data'
    parser_base = construct_dataReader(get_output_file(base_size), 'standard')
    parser_9000 = construct_dataReader(get_output_file(base_size, 9000), 'standard')
    parser_18000 = construct_dataReader(get_output_file(base_size, 18000), 'standard')

    X_base, y_base = parser_base.read_format_data(feature)
    X_9000, y_9000 = parser_9000.read_format_data(feature)
    X_18000, y_18000 = parser_18000.read_format_data(feature)

    if cross_validation:
        X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(X_base, y_base, test_size=CV_proportion)
        X_9000_train, X_9000_test, y_9000_train, y_9000_test = train_test_split(X_9000, y_9000, test_size=CV_proportion)
        X_18000_train, X_18000_test, y_18000_train, y_18000_test = train_test_split(X_18000, y_18000, test_size=CV_proportion)

        dataset = {
            0: (X_base_train, X_base_test, y_base_train, y_base_test, parser_base.relations),
            9000: (X_9000_train, X_9000_test, y_9000_train, y_9000_test, parser_9000.relations),
            18000: (X_18000_train, X_18000_test, y_18000_train, y_18000_test, parser_18000.relations)
        }
    else:
        parser_test = construct_dataReader(test_file, 'Angli')
        X_test, y_test = parser_test.read_format_data(feature)

        dataset = {
            0: (X_base, X_test, y_base, y_test, parser_base.relations),
            9000: (X_9000, X_test, y_9000, y_test, parser_9000.relations),
            18000: (X_18000, X_test, y_18000, y_test, parser_18000.relations)
        }

    return dataset


def train_models(dataset, feature, base_size):
    classifier = build_model(get_output_file(base_size), dataset[0][0], dataset[0][2], feature)
    classifier_9000 = build_model(get_output_file(base_size, 9000), dataset[9000][0], dataset[9000][2], feature)
    classifier_18000 = build_model(get_output_file(base_size, 18000), dataset[18000][0], dataset[18000][2], feature)

    return (classifier, classifier_9000, classifier_18000)


def build_model(input_file, X, y, feature):
    """
    Read training data from input file, fit a classifier model according to the training data.
    Read testing data from test file, and test the classifier model.
    """
    print "training on %s" % input_file

    # my_classifier = LinearClassifier(feature=feature)
    my_classifier = OneVsRestClassifier(feature=feature)
    my_classifier.fit(X, y)
    return my_classifier


def test_models(classifiers, dataset, base_size):
    """
    test_sets = {
        0: (classifier_base, X_base_test, y_base_test),
        ...
    }
    """

    print "testing classifiers"

    test_set = {
        0: (classifiers[0], dataset[0][1], dataset[0][3]),
        9000: (classifiers[1], dataset[9000][1], dataset[9000][3]),
        18000: (classifiers[2], dataset[18000][1], dataset[18000][3])
    }

    draw_f1_curve(test_set, base_size)

    if False:
        (confusion_table, accuracy, precision_recall, f1_micro, f1_macro) = model_test(classifiers[0], test_set[0][1],
                                                                                             test_set[0][2], dataset[0][4])
        generate_report('semantic', 'logit', '0_report.txt', confusion_table, accuracy, precision_recall, f1_micro, f1_macro, dataset[0][4])

        (confusion_table, accuracy, precision_recall, f1_micro, f1_macro) = model_test(classifiers[1], test_set[9000][1],
                                                                                             test_set[9000][2], dataset[9000][4])
        generate_report('semantic', 'logit', '9000_report.txt', confusion_table, accuracy, precision_recall, f1_micro, f1_macro, dataset[9000][4])


        (confusion_table, accuracy, precision_recall, f1_micro, f1_macro) = model_test(classifiers[2], test_set[18000][1],
                                                                                             test_set[18000][2], dataset[18000][4])
        generate_report('semantic', 'logit', '18000_report.txt', confusion_table, accuracy, precision_recall, f1_micro, f1_macro, dataset[18000][4])


if __name__ == '__main__':
    opt = read_command()

    sizes = ['all']
    if opt.input:
        create_mixture_data(sizes)
    else:
        feature = 'semantic'
        for base_size in sizes:
            dataset = get_data(base_size, 'test', feature, cross_validation=False)
            classifier, classifier_9000, classifier_18000 = train_models(dataset, feature, base_size=base_size)
            test_models([classifier, classifier_9000, classifier_18000], dataset, base_size=base_size)
