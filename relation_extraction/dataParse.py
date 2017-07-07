import re
import sys
import ast


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

    def read_format_data(self, feature='unigram'):
        """
        Read data from input file, handles out-of-format data if there's any, and parse the data.

        :return: parsed data
        """
        (data, abnormal_data, relations) = self.read_data()
        self.relations = relations

        if len(abnormal_data) > 0:
            self.handle_abnormal_data(abnormal_data)
        return self.format_data(data, feature)

    def read_data(self):
        """
        Read data from input file.

        :return: data information, abnormal data instances, set of all relations
        """
        raise NotImplementedError

    def format_data(self, data, feature):
        """
        Parse data information and store all the instances sentences into X,
        all the corresponding relation labels into y.

        :param data: data information

        :return: parsed data
        """
        X = []
        y = []
        for instance in data:
            if feature == 'unigram':
                X.append(instance['sentence'])
            elif feature == 'semantic':
                X.append(instance['features'])
            else:
                print "parser unknown feature type %s" % feature

            y.append(instance['relation'])
        return X, y

    def handle_abnormal_data(self, abnormal_data):
        """
        Handle out-of-format data.

        :param abnormal_data: out-of-format data instances
        """
        raise NotImplementedError


class SemEvalReader(DataReader):

    def read_data(self):
        """
        data = [
            ...,
            {
                e1: (entity 1, start index, end index),
                e2: (entity 2, start index, end index),
                sentence: instance sentence,
                relation: relation,
                source: self.input_file,
                feature:
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

                e1 = sentence[sentence.index('<e1>') + 4: sentence.index('</e1>')]
                e2 = sentence[sentence.index('<e2>') + 4: sentence.index('</e2>')]
                e1_start = sentence.index('<e1>')
                e2_start = sentence.index('<e2>')
                if e1_start < e2_start:
                    instance['e1'] = ( e1, sentence.index('<e1>'), sentence.index('</e1>') - 4 )
                    instance['e2'] = ( e2, sentence.index('<e2>') - 9, sentence.index('</e2>') - 13 )
                else:
                    instance['e1'] = ( e1, sentence.index('<e1>') - 9, sentence.index('</e1>') - 13 )
                    instance['e2'] = ( e1, sentence.index('<e2>'), sentence.index('</e2>') - 4 )

                # instance['e1'] = ( sentence[sentence.index('<e1>') + 4: sentence.index('</e1>')], \
                #                  sentence.index('<e1>'), sentence.index('</e1>') - 5 )
                # instance['e2'] = ( sentence[sentence.index('<e2>') + 4: sentence.index('</e2>')], \
                #                  sentence.index('<e2>'), sentence.index('</e2>') - 5 )

                sentence = re.sub("<e1>|</e1>|<e2>|</e2>", "", sentence)
                instance['sentence'] = sentence

                if relationInfo == "Other":
                    if relationInfo not in relations:
                        relations.append(relationInfo)
                    instance['relation'] = [relationInfo]
                    data.append(instance)
                else:
                    relation = relationInfo[0: relationInfo.index('(')]
                    if relation not in relations:
                        relations.append(relation)
                    instance['relation'] = [relation]
                    data.append(instance)
                    direction = relationInfo[relationInfo.index('(') + 1: relationInfo.index(')')]
                    if direction == 'e2, e1':
                        instance['e1'], instance['e2'] = instance['e2'], instance['e1']

                instance['source'] = self.input_file

            except StopIteration:
                break

            except ValueError:
                abnormalData.append(sentence)

        return data, abnormalData, relations

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


class NaaclReader(DataReader):

    def read_data(self):
        data = []
        abnormalData = []
        relations = []

        with open(self.input_file, 'r') as file:
            content = file.readlines()

        for row in content:
            try:
                instance = {}

                instance_info = row.split('\t')


                instance['e1'] = (instance_info[0], instance_info[1], instance_info[2])
                instance['e2'] = (instance_info[3], instance_info[4], instance_info[5])

                instance['relation'] = [instance_info[7]]
                if instance_info[7] == 'NA':
                    instance['relation'] = []

                if instance_info[7] not in relations:
                    relations.append(instance_info[7])

                instance['sentence'] = instance_info[8]

                instance['features'] = []
                for i in range(9, len(instance_info)):
                    instance['features'].append(instance_info[i].rstrip())

                instance['source'] = self.input_file

                data.append(instance)

            except StopIteration:
                break

            except Exception:
                abnormalData.append(row)

        return data, abnormalData, relations

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


class StandardReader(DataReader):

    def read_data(self):
        data = []
        abnormalData = []
        relations = []

        with open(self.input_file, 'r') as file:
            content = file.readlines()

        for row in content:
            try:
                instance = {}

                instance_info = row.split('\t')

                instance['source'] = instance_info[0]

                instance['e1'] = (instance_info[1], instance_info[2], instance_info[3])
                instance['e2'] = (instance_info[4], instance_info[5], instance_info[6])

                instance['relation'] = ast.literal_eval(instance_info[7])
                for rel in instance['relation']:
                    if rel not in relations:
                        relations.append(rel)

                # instace['relation'] = instance_info[7]
                # if instance_info[7] not in relations:
                #     relations.append(instance_info[7])

                instance['sentence'] = instance_info[8]

                instance['features'] = []
                for i in range(8, len(instance_info)):
                    instance['features'].append(instance_info[i].rstrip())

                data.append(instance)

            except StopIteration:
                break

            except ValueError:
                abnormalData.append(row)

        return data, abnormalData, relations

    def handle_abnormal_data(self, abnormal_data):
        pass


class AngliReader(DataReader):
    def read_data(self):
        data = []
        abnormalData = []
        relations = []

        with open(self.input_file, 'r') as file:
            content = file.readlines()

        for row in content:
            try:
                instance = {}

                instance_info = row.rstrip().split('\t')

                instance = self._read_line_angli(instance_info)

                instance['source'] = self.input_file

                for rel in instance['relation']:
                    if rel not in relations:
                        relations.append(rel)

                data.append(instance)

            except StopIteration:
                break

            except ValueError:
                abnormalData.append(row)

        return data, abnormalData, relations

    def _read_line_angli(self, vals, annotator='soderland'):
        """Read line from data/gold1/test_multiPositive.tsv format."""
        def to_val(label):
            """Return numerical value based on Angli-style label"""
            if label == 'optional':
                return 0
            elif label.endswith('neg'):
                return -1
            else:
                return 1
        angli_relation_types = ['has nationality', 'born in',
                                'lived in', 'died in', 'traveled to']
        normal_types = ['per:origin', '/people/person/place_of_birth', '/people/person/place_lived', '/people/deceased_person/place_of_death', 'travel']
        
        relations = dict((k, to_val(v)) for k, v in zip(normal_types, ast.literal_eval(vals[7])))

        instance = {}
        relation = [] 
        for rel in relations:
            if relations[rel] == 1:
                relation.append(rel)
        instance['relation'] = relation

        d = {'arg0': vals[0],
             'i00': int(vals[1]),
             'i01': int(vals[2]),
             'arg1': vals[3],
             'i10': int(vals[4]),
             'i11': int(vals[5]),
             'id': vals[6],
             'relations': relations,
             'i0': int(vals[9]),
             'i1': int(vals[10]),
             'txt': vals[11]}
        # Populate indices for travel docs, since they contain
        # no indices in the file from Angli.
        if d['id'].startswith('travel'):
            d['i0'] = 0  # This is true anyways, but let's guarantee it.
            d['i00'] = d['txt'].find(d['arg0'])
            d['i10'] = d['txt'].find(d['arg1'])
        # Guarantee that entity indice ranges match the length of the entity.
        d['i01'] = d['i00'] + len(d['arg0'])
        d['i11'] = d['i10'] + len(d['arg1'])
        # Guarantee that excerpt indice ranges match length.
        d['i1'] = d['i0'] + len(d['txt'])

        instance['e1'] = (d['arg0'], d['i00'], d['i01'])
        instance['e2'] = (d['arg1'], d['i10'], d['i11'])

        instance['sentence'] = vals[11]

        instance['features'] = []
        for i in range(12, len(vals)):
            instance['features'].append(vals[i].rstrip())

        return instance

def construct_dataReader(file, format):
    if format == 'SemEval':
        return SemEvalReader(file)
    elif format == 'Naacl':
        return NaaclReader(file)
    elif format == 'standard':
        return StandardReader(file)
    elif format == 'Angli':
        return AngliReader(file)
    else:
        print "Unknown file format", format
        sys.exit()
