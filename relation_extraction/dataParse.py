import re
import sys


class DataReader:

    def __init__(self, input_file):
        self.input_file = input_file

    def read_format_data(self):
        raise NotImplementedError

    def read_data(self):
        raise NotImplementedError

    def format_data(self, data):
        raise  NotImplementedError

    def handle_abnormal_data(self, abnormal_data):
        raise  NotImplementedError


class SemEvalReader(DataReader):

    def read_format_data(self):
        (data, abnormal_data) = self.read_data()
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
                sys.exit(0)
            elif action == 'p':
                for sentence in abnormal_data:
                    print sentence
            elif action == '':
                return
            else:
                print "unknown command"

