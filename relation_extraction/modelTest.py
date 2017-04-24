def model_test(my_classifier, X_test, y):
    result = my_classifier.predict(X_test)
    confusion_table = get_confusion_table(result, y)
    precision_recall = get_precision_recall(confusion_table)
    return confusion_table, precision_recall


def get_confusion_table(result, y):
    confusion_table = {}
    for i, relation in enumerate(y):
        if relation not in confusion_table:
            confusion_table[relation] = {}
        relation_pred = result[i]
        if relation_pred not in confusion_table[relation]:
            confusion_table[relation][relation_pred] = 0
        confusion_table[relation][relation_pred] += 1
    return confusion_table


def get_accuracy(result, y):
    good = 0
    for i, pred in enumerate(result):
        if pred == y[i]:
            good += 1


def get_precision_recall(confusion_table):
    precision_recall = {}
    relations = confusion_table.keys()
    for target_relation in relations:
        # true_positive = confusion_table[target_relation][target_relation]
        true_positive = get_num_pred(confusion_table, target_relation, target_relation)
        num_target_true = 0
        num_target_pred = 0
        for relation in relations:
            # num_target_true += confusion_table[target_relation][relation]
            # num_target_pred += confusion_table[relation][target_relation]

            num_target_true += get_num_pred(confusion_table, target_relation, relation)
            num_target_pred += get_num_pred(confusion_table, relation, target_relation)

        precision = (true_positive + 0.0) / num_target_pred
        recall = (true_positive + 0.0) / num_target_true

        precision_recall[target_relation] = (precision, recall)

    return precision_recall


def get_num_pred(confusion_table, target, pred):
    predictions = confusion_table[target]
    if pred in predictions:
        return predictions[pred]
    else:
        return 0
