import re


def read_data(input_file):
    """
    data = [
        ...,
        {
            e1: entity 1,
            e2: entity 2,
            direction: (e2, e1),
            sentence: instance sentence,
            relation: relation
        },
        ...
    ]
    """
    data = []
    abnormalData = []

    with open(input_file, 'r') as file:
        content = file.readlines()

    it = iter(content)
    while True:
        try:
            instance = {}

            line = it.next().strip()
            relationInfo = it.next().strip()
            it.next()
            it.next()

            sentence = line[line.index('\"') + 1: -1]
            instance['e1'] = sentence[sentence.index('<e1>') + 4: sentence.index('</e1>')]
            instance['e2'] = sentence[sentence.index('<e2>') + 4: sentence.index('</e2>')]

            sentence = re.sub("<e1>|</e1>|<e2>|</e2>", "", sentence)
            instance['sentence'] = sentence

            if relationInfo == "Other":
                instance['relation'] = relationInfo
                data.append(instance)
            else:
                relation = relationInfo[0: relationInfo.index('(')]
                instance['relation'] = relation
                data.append(instance)
                direction = relationInfo[relationInfo.index('(') + 1: relationInfo.index(')')]
                instance['direction'] = (direction[0: 2], direction[3: 5])


        except StopIteration:
            break

        except ValueError:
            abnormalData.append(sentence)

    return data, abnormalData


def format_data(data):
    '''
    Parse the data and store all the instances sentences into X, all the corresponding relation labels into y.
    '''
    X = []
    y = []
    for instance in data:
        X.append(instance['sentence'])
        y.append(instance['relation'])
    return X, y
