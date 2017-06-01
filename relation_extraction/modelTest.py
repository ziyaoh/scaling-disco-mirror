from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def draw_f1_curve(test_sets):
    f1s_micro = []
    f1s_macro = []
    training_data_labels = []
    # report = PdfPages('mixture_report.pdf')

    for label in test_sets:
        training_data_labels.append(label)

        classifier, X_test, y_test = test_sets[label]
        y_pred = classifier.predict(X_test)
        f1s_micro.append(get_f1_score(y_pred, y_test, 'micro'))
        f1s_macro.append(get_f1_score(y_pred, y_test, 'macro'))

    plt.plot(training_data_labels, f1s_micro)
    plt.plot(training_data_labels, f1s_macro)
    plt.legend(['micro', 'macro'], loc='upper left')
    plt.show()


def get_f1_score(y_pred, y_test, average):
    return f1_score(y_test, y_pred, average=average)


def model_test(my_classifier, X_test, y_test, relations):
    """
    Test classifier by predicting labels for each instance in X_test,
    and comparing the predictions with their actual labels.
    :param my_classifier: tested classifier
    :param X_test: testing data
    :param y_test: testing labels
    :return: confusion_table, prediction accuracy, and precision, recall for each class
    """
    y_pred = my_classifier.predict(X_test)
    confusion_table = get_confusion_table(y_pred, y_test, relations)
    accuracy = get_accuracy(y_pred, y_test)
    precision_recall = get_precision_recall(confusion_table)
    f1_micro = get_f1_score(y_pred, y_test, 'micro')
    f1_macro = get_f1_score(y_pred, y_test, 'macro')
    return confusion_table, accuracy, precision_recall, f1_micro, f1_macro


def get_confusion_table(y_pred, y_test, relations):
    """
    Get the confusion table.
    :param y_pred:
    :param y_test:
    :return: {actual_class1: {pred_class1: number, pred_class2: number, ...}, actual_class2, ...}
    """
    confusion_table = {}
    for actual_relation in relations:
        confusion_table[actual_relation] = {}
        for pred_relation in relations:
            confusion_table[actual_relation][pred_relation] = 0

    for i, relation in enumerate(y_test):
        relation_pred = y_pred[i]
        confusion_table[relation][relation_pred] += 1
    return confusion_table


def get_accuracy(y_pred, y_test):
    """
    Get the prediction accuracy, which is number of correct predictions / number of all predictions.
    :param y_pred:
    :param y_test:
    :return: prediction accuracy
    """
    good = 0
    for i, pred in enumerate(y_pred):
        if pred == y_test[i]:
            good += 1
    return (good + 0.0) / len(y_pred)


def get_precision_recall(confusion_table):
    """
    Get precision and recall for each class according to confusion table.
    :param confusion_table:
    :return: {class1: (precision, recall), class2, ...}
    """
    precision_recall = {}
    relations = confusion_table.keys()
    for target_relation in relations:
        # true_positive = get_num_pred(confusion_table, target_relation, target_relation)
        true_positive  = confusion_table[target_relation][target_relation]
        num_target_true = 0
        num_target_pred = 0

        for relation in relations:
            # num_target_true += get_num_pred(confusion_table, target_relation, relation)
            # num_target_pred += get_num_pred(confusion_table, relation, target_relation)

            num_target_pred += confusion_table[relation][target_relation]
            num_target_true += confusion_table[target_relation][relation]

        if true_positive == 0:
            precision = 0
            recall = 0
        else:
            precision = (true_positive + 0.0) / num_target_pred
            recall = (true_positive + 0.0) / num_target_true

        # if target_relation == '/people/person/place_lived':
        #     print '/people/person/place_lived', true_positive, num_target_true, num_target_pred

        precision_recall[target_relation] = (precision, recall)

    return precision_recall


def get_num_pred(confusion_table, actual_class, pred_class):
    """
    Helper function to get the number of instances that are actual_class
    but predicted as pred_class from confusion table.
    :param confusion_table:
    :param actual_class:
    :param pred_class:
    """
    if actual_class not in confusion_table:
        return 0

    predictions = confusion_table[actual_class]
    if pred_class in predictions:
        return predictions[pred_class]
    else:
        return 0
