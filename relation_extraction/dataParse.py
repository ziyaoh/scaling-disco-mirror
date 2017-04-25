import re
import sys


class DataReader:
    """
    A DataReader handles works related with reading and parsing input data.
    """

    def __init__(self, input_file):
        """
        Initializer for a DataReader object.

        :param input_file: input file name
        """
        self.input_file = input_file

    def read_format_data(self):
        """
        Read data from input file, handles out-of-format data if there's any, and parse the data.

        :return: parsed data
        """
        raise NotImplementedError

    def read_data(self):
        """
        Read data from input file.

        :return: data information, abnormal data instances, set of all relations
        """
        raise NotImplementedError

    def format_data(self, data):
        """
        Parse data informatino.

        :param data: data information

        :return: parsed data
        """
        raise  NotImplementedError

    def handle_abnormal_data(self, abnormal_data):
        """
        Handle out-of-format data.

        :param abnormal_data: out-of-format data instances
        """
        raise NotImplementedError


class SemEvalReader(DataReader):

    def read_format_data(self):
        (data, abnormal_data, relations) = self.read_data()
        self.relations = relations

        if len(abnormal_data) > 0:
            self.handle_abnormal_data(abnormal_data)
        return self.format_data(data)

    def read_data(self):
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
        relations = []

        with open(self.input_file, 'r') as file:
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
                    if relationInfo not in relations:
                        relations.append(relationInfo)
                    instance['relation'] = relationInfo
                    data.append(instance)
                else:
                    relation = relationInfo[0: relationInfo.index('(')]
                    if relation not in relations:
                        relations.append(relation)
                    instance['relation'] = relation
                    data.append(instance)
                    direction = relationInfo[relationInfo.index('(') + 1: relationInfo.index(')')]
                    instance['direction'] = (direction[0: 2], direction[3: 5])


            except StopIteration:
                break

            except ValueError:
                abnormalData.append(sentence)

        return data, abnormalData, relations

    def format_data(self, data):
        '''
        Parse the data and store all the instances sentences into X, all the corresponding relation labels into y.
        '''
        X = []
        y = []
        for instance in data:
            X.append(instance['sentence'])
            y.append(instance['relation'])
        return X, y

    def handle_abnormal_data(self, abnormal_data):
        print "number of abnormal data instances in", self.input_file, ":", len(abnormal_data)
        while True:
            action = raw_input("'q' to exit, 'p' to print the abnormal data instances, Enter to continue")
            if action == 'q':
                sys.exit()
            elif action == 'p':
                for sentence in abnormal_data:
                    print sentence
            elif action == '':
                return
            else:
                print "unknown command"

