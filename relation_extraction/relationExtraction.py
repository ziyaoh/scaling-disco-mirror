import argparse
import sys
import classifier
import modelTest
from contextlib import contextmanager
from dataParse import SemEvalReader


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
                        default='logit',
                        choices=['logit', 'NN'],
                        help='The type of classifier we want to use. If not specified, logistic regression classifier '
                             'will be used.')
    parser.add_argument('-o', '--output',
                        default=None,
                        help='The testing report file name. If not specified, report will be printed to standard out.')
    parser.add_argument('-d', '--dataformat',
                        default='SemEval',
                        choices=['SemEval'],
                        help='The input file data format. We will parse input file according to this information. '
                             'If not specified, SemEval format will be used.')

    return parser.parse_args()


def build_model(input_file, data_format):
    """
    Read training data from input file, fit a classifier model according to the training data.
    Read testing data from test file, and test the classifier model.
    """
    if data_format == 'SemEval':
        parser = SemEvalReader(input_file)
    else:
        print 'Unknown input format.'
        sys.exit()

    (X, y) = parser.read_format_data()

    my_classifier = classifier.LinearClassifier(opt.feature, opt.classifier)
    my_classifier.fit(X, y)
    return my_classifier, parser.relations


def test_model(my_classifier, test_file, output_file, data_format, relations):
    """
    Test the classifier on testing data, and return resulting confusion table, prediction accuracy, precision_recall and
    all relations in testing data.
    """
    if data_format == 'SemEval':
        parser = SemEvalReader(test_file)
    else:
        print 'Unknown input format.'
        sys.exit()

    (X_test, y_test) = parser.read_format_data()

    (confusion_table, accuracy, precision_recall) = modelTest.model_test(my_classifier, X_test, y_test, relations)
    return (confusion_table, accuracy, precision_recall)


def generate_report(feature_type, classifier_type, output_file, confusion_table, accuracy, precision_recall):
    """
    Generate a report based on testing data.
    """
    with report_writer(output_file) as writer:
        writer.write("Report for %s feature %s classifier:\n" % (feature_type, classifier_type))

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

        writer.write("\nPrecision and Recall:\n")
        writer.write("".rjust(width) + "precision".rjust(10) + "recall".rjust(10))
        for relation in relations:
            p_r = precision_recall[relation]
            writer.write("\n" + relation.rjust(width) + "%.4f".rjust(10) % p_r[0] + "%.4f".rjust(10) % p_r[1])
        writer.write("\n")



@contextmanager
def report_writer(file_name):
    """
    Helper function for generating report. Return an output stream to file if filename is specified,
    return stdout otherwise.
    """
    if file_name is None:
        yield sys.stdout
    else:
        with open(file_name, 'w') as out_file:
            yield out_file


if __name__ == '__main__':
    opt = read_command()
    (my_classifier, relations) = build_model(opt.input, opt.dataformat)
    (confusion_table, accuracy, precision_recall) \
        = test_model(my_classifier, opt.test, opt.output, opt.dataformat, relations)
    generate_report(opt.feature, opt.classifier, opt.output, confusion_table, accuracy, precision_recall)
