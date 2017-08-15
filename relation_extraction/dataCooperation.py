import argparse
import random
import sys
import os
from dataParse import construct_dataReader

def read_command():
    """
    main_file: the main data file
    main_format: format of the main data file
    --inputs: additional input data files
    --formats: formats of additional input data files
    --proportions: number of instances that should be used from each additional input file
    --output: output file name
    """
    parser = argparse.ArgumentParser()

    # parser.add_argument('main_file',
    #                     help='The main input file name.')
    # parser.add_argument('main_format',
    #                     help='Format of the main input file.')
    parser.add_argument('-i', '--inputs',
                        nargs='+',
                        required=True,
                        help='Input files. Number of instances taken from each ')
    parser.add_argument('-f', '--formats',
                        nargs='+',
                        required=True,
                        help='Format of each input file data. Must be in one to one correspondence with input files')
    parser.add_argument('-p', '--proportions',
                        nargs='+',
                        default=None,
                        help='Number of instances that should be used from each additional input file. If not '
                             'specified, all instances of all additional input files will be used. To use all instances'
                             ' of a specific input file, specify \'all\' as the corresponding argument. Each argument '
                             'must be an int or the string \'all\'.')
    parser.add_argument('-o', '--output',
                        default=None,
                        help='Final output data file name.')
    parser.add_argument('-n', '--neg-proportion',
                        default=None,
                        help='Proportion between number of negative samples and positive samples in final training data')

    opt = parser.parse_args()

    # if opt.inputs:
    #     if not opt.formats:
    #         parser.error('with --inputs specified, --formats is required')
    #     elif not len(opt.inputs) == len(opt.formats):
    #         parser.error('--inputs should have same number of elements with --formats')
    #     elif opt.proportions and not len(opt.inputs) == len(opt.proportions):
    #         parser.error('--inputs should have same number of elements with --proportions')
    # elif opt.formats or opt.proportions:
    #     parser.error('without --inputs, --formats and --proportions are not allowed')

    if not len(opt.inputs) == len(opt.formats):
        parser.error('--inputs should have same number of elements with --formats')
    elif opt.proportions and not len(opt.inputs) == len(opt.proportions):
        parser.error('--inputs should have same number of elements with --proportions')

    if opt.proportions:
        for i, prop in enumerate(opt.proportions):
            try:
                int_prop = int(prop)
                opt.proportions[i] = int_prop
            except ValueError:
                if not prop == 'all':
                    print "unknown proportion argument %s" % prop
                    sys.exit()
                else:
                    parser = construct_dataReader(opt.inputs[i], opt.formats[i])
                    opt.proportions[i] = len(parser.read_data()[0])

    if opt.neg_proportion:
        try:
            opt.neg_proportion = int(opt.neg_proportion)
        except ValueError:
            print "unknown neg_proportion argument %s" % opt.neg_proportion
            sys.exit()

    return opt


def merge_DS_CS(DS, DSFormat, DSProportion, CS, CSFormat, CSProportion, negProportion):
    posDS = 'posDS'
    negDS = 'negDS'
    posCS = 'posCS'
    negCS = 'negCS'

    numDSPos, numDSNeg = split_pos_neg(DS, DSFormat, posDS, negDS)
    numCSPos, numCSNeg = split_pos_neg(CS, CSFormat, posCS, negCS)

    staticTrainingFile = 'staticTrainingFile'
    scalingTrainingFile = 'scalingTrainFile'

    numDSPosUse = int(round(DSProportion / (1 + negProportion)))
    numCSPosUse = int(round(CSProportion / (1 + negProportion)))
    numDSPosUse = numDSPos if numDSPos <= numDSPosUse else numDSPosUse
    numCSPosUse = numCSPos if numCSPos <= numCSPosUse else numCSPosUse

    staticTrainingData = cooperate_data([posCS, negCS], ['standard', 'standard'], [numCSPosUse, negProportion * numCSPosUse])
    scalingTrainingData = cooperate_data([posDS, negDS], ['standard', 'standard'], [numDSPosUse, negProportion * numDSPosUse])

    trainingData = staticTrainingData + scalingTrainingData
    random.shuffle(trainingData)

    os.remove(posDS)
    os.remove(negDS)
    os.remove(posCS)
    os.remove(negCS)
    
    return trainingData


def split_pos_neg(dataFile, dataFormat, posFile, negFile):
    numPos = 0
    numNeg = 0
    with open(posFile, 'w') as posWriter, open(negFile, 'w') as negWriter:
        data = construct_dataReader(dataFile, dataFormat).read_data()[0]
        for instance in data:
            if len(instance['relation']) == 0:
                write_instance(instance, negWriter)
                numNeg += 1
            else:
                write_instance(instance, posWriter)
                numPos += 1
    return numPos, numNeg


def cooperate_data(files, formats, proportions=None):
    """
    Cooperate data from different files together.
    :return: list of data instances
    """
    # main_parser = construct_dataReader(main_file, main_format)
    # final_data = main_parser.read_data()[0]
    #
    # if not files:
    #     return final_data

    parsers = []
    final_data = []
    for i, file in enumerate(files):
        format = formats[i]
        parsers.append(construct_dataReader(file, format))

    if proportions:
        for i, reader in enumerate(parsers):
            data = reader.read_data()[0]
            if type(proportions[i]) is str:
                final_data.extend(data)
            else:
                # in case user requires more instances than a file contains
                try:
                    sample = random.sample(data, proportions[i])
                    final_data.extend(sample)
                except ValueError:
                    print "%s does not contain enough data instances" % files[i]
                    print "Total number of data instances: %s" % len(data)
                    while True:
                        action = raw_input("please reenter the number of instances taken from this file, "
                                           "or 's' for skip, 'q' for quit")
                        try:
                            num = int(action)
                            if num > len(data):
                                continue
                            sample = random.sample(data, num)
                            final_data.extend(sample)
                            break
                        except ValueError:
                            if action == 's':
                                break
                            elif action == 'q':
                                sys.exit()
                            else:
                                print "Unknown command"

    else:
        for reader in parsers:
            final_data.extend(reader.read_data()[0])

    random.shuffle(final_data)
    return final_data


def write_final_data(final_data, output_file='test_output', output_format='standard'):
    """
    e1  start_index1    end_index1  e2  start_index2    end_index2  relation    sentence    source
    """
    if output_format == 'standard':
        with open(output_file, 'w') as writer:
            for _, instance in enumerate(final_data):
                write_instance(instance, writer)

    elif output_format == 'datum':
        with open(output_file, 'w') as file:
            for _, instance in enumerate(final_data):
                # only consider single relation instance for now
                rel = '_NR' if len(instance['relation']) == 0 else instance['relation'][0]
                file.write('{%s} PER GEO %s %s ' % (instance['e1'][0], instance['e2'][0], rel))
                if 'features' in instance:
                    for i, feature in enumerate(instance['features']):
                        file.write('%s ' % feature)
                file.write('\n')


def write_instance(instance, writer):
    writer.write('%s\t' % instance['source'])
    writer.write('%s\t%s\t%s\t' % (instance['e1'][0], instance['e1'][1], instance['e1'][2]))
    writer.write('%s\t%s\t%s\t' % (instance['e2'][0], instance['e2'][1], instance['e2'][2]))
    writer.write('%s\t' % instance['relation'])
    writer.write('%s\t' % instance['sentence'])
    if 'features' in instance:
        for i, feature in enumerate(instance['features']):
            writer.write('%s' % feature)
            if i < len(instance['features']) - 1:
                writer.write('\t')
    writer.write('\n')


if __name__ == '__main__':
    opt = read_command()

    if opt.neg_proportion:
        if len(opt.inputs) != 2:
            print 'With neg_proportion specified, input has to be two files, in the order of [DSFile, CSFile]'
            sys.exit()
        final_data = merge_DS_CS(opt.inputs[0], opt.formats[0], opt.proportions[0], opt.inputs[1], opt.formats[1], opt.proportions[1], opt.neg_proportion)
    else:
        final_data = cooperate_data(opt.inputs, opt.formats, opt.proportions)
    if opt.output:
        output = opt.output
    else:
        output = '%s_%s_%s_%s' % (opt.inputs[0], opt.proportions[0], opt.inputs[1], opt.proportions[1])
        if opt.neg_proportion:
            output += 'neg_%s' % opt.neg_proportion
    write_final_data(final_data, output)
