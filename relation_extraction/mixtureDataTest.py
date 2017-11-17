import os
import sys
import argparse
from relationExtraction import parse_data, build_model, test_model
from dataParse import construct_dataReader
from sklearn.model_selection import train_test_split


base_file = 'train_DS'
input_9000 = 'DS_random_9000'
input_18000 = 'DS_random_18000'

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
        output_file = "DS_%s_neg_1" % base_size
    else:
        output_file = "DS_%s_CS_%s_neg_1" % (base_size, additional_size)
    return output_file


def get_data(base_size, test_file, feature, classifier_type, cross_validation=False):
    if feature != 'unigram' and feature != 'semantic':
        print 'unknown feature type'
        sys.exit()

    print 'reading data'

    X_base, y_base, relations_base, sig_base = parse_data(get_output_file(base_size), 'standard', feature, classifier_type)
    X_9000, y_9000, relations_9000, sig_9000 = parse_data(get_output_file(base_size, 9000), 'standard', feature, classifier_type)
    X_18000, y_18000, relations_18000, sig_18000 = parse_data(get_output_file(base_size, 18000), 'standard', feature, classifier_type)

    if cross_validation:
        X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(X_base, y_base, test_size=CV_proportion)
        X_9000_train, X_9000_test, y_9000_train, y_9000_test = train_test_split(X_9000, y_9000, test_size=CV_proportion)
        X_18000_train, X_18000_test, y_18000_train, y_18000_test = train_test_split(X_18000, y_18000, test_size=CV_proportion)

        dataset = {
            0: (X_base_train, X_base_test, y_base_train, y_base_test, relations_base),
            9000: (X_9000_train, X_9000_test, y_9000_train, y_9000_test, relations_9000),
            18000: (X_18000_train, X_18000_test, y_18000_train, y_18000_test, relations_18000)
        }
    else:
        X_test, y_test, relations_test, sig_test = parse_data(test_file, 'standard', feature, classifier_type)

        dataset = {
            0: (X_base, X_test, y_base, y_test, relations_base),
            9000: (X_9000, X_test, y_9000, y_test, relations_9000),
            18000: (X_18000, X_test, y_18000, y_test, relations_18000)
        }

    return dataset


def train_models(dataset, feature, classifier_type):
    classifier = build_model(dataset[0][0], dataset[0][2], feature, classifier_type, get_output_file(base_size))
    classifier_9000 = build_model(dataset[9000][0], dataset[9000][2], feature, classifier_type, get_output_file(base_size, 9000))
    classifier_18000 = build_model(dataset[18000][0], dataset[18000][2], feature, classifier_type, get_output_file(base_size, 18000))

    return (classifier, classifier_9000, classifier_18000)


def test_models(classifiers, dataset, base_size):
    """
    test_sets = {
        0: (classifier_base, X_base_test, y_base_test),
        ...
    }
    """

    print "testing classifiers"

    test_model(classifiers[0], dataset[0][1], dataset[0][3], '%s.score' % get_output_file(base_size))
    test_model(classifiers[1], dataset[9000][1], dataset[9000][3], '%s.score' % get_output_file(base_size, 9000))
    test_model(classifiers[2], dataset[18000][1], dataset[18000][3], '%s.score' % get_output_file(base_size, 18000))


if __name__ == '__main__':
    opt = read_command()

    sizes = ['all']

    if opt.input:
        create_mixture_data(sizes)
    else:
        feature = 'semantic'
        for base_size in sizes:
            dataset = get_data(base_size, 'new_test', feature, 'OneVsRest', cross_validation=False)
            classifier, classifier_9000, classifier_18000 = train_models(dataset, feature, 'OneVsRest')
            test_models([classifier, classifier_9000, classifier_18000], dataset, base_size=base_size)
